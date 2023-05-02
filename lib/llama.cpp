#include "llama.hpp"
#include "file_reader.hpp"
#include <cassert>
#include "macro.hpp"
#include <fstream>
#include <numeric>
#include <functional>
#include "span.hpp"
#include <cstring>
#include <cmath>
#include <thread>
#include <mutex>
#include <optional>
#include "file_loader.hpp"
#include "utils.hpp"

namespace fastllama {

    auto Model::unload() -> void {
        is_valid = false;
        kv_self.deinit();
    }

    bool KVCacheBuffer::init(HyperParams const& params, Logger const& logger) {
        using namespace ::fastllama::literals;

        auto const n_ctx  = params.n_ctx;
        auto const n_embd  = params.n_embd;
        auto const n_layer = params.n_layer;

        std::size_t mem_size = static_cast<std::size_t>(n_layer * n_ctx);
        std::size_t number_of_elements = static_cast<std::size_t>(n_embd) * mem_size;
        auto const buffer_size = 2 * number_of_elements * static_cast<std::size_t>(ggml_type_size(memory_type)) + 2_MiB;
        buffer.resize( buffer_size );

        this->ctx = MemContext(buffer.data(), buffer.size(), false);

        if (!this->ctx) {
            logger.log_err("KVCacheBuffer::init", "failed to allocate memory for kv cache\n");
            return false;
        }

        this->k = ggml_new_tensor_1d(this->ctx.get(), memory_type, static_cast<std::int64_t>(number_of_elements));
        this->v = ggml_new_tensor_1d(this->ctx.get(), memory_type, static_cast<std::int64_t>(number_of_elements));

        auto const total_kv_size = ggml_nbytes(this->k) + ggml_nbytes(this->v);

        logger.log("KVCacheBuffer::init", "kv self size  = ", dyn_humanize_size(total_kv_size), "\n");
        
        return true;
    }

    void KVCacheBuffer::deinit([[maybe_unused]] Logger const& logger) {
        ctx.free();
    }

    bool KVCacheBuffer::save_state(BinaryFileWriter& writer, Logger const& logger) const noexcept {
        writer.write(&memory_type);


        logger.log(__func__, "saving key cache\n");
        writer.write(k->data, sizeof(char), ggml_nbytes(k));
        
        logger.log(__func__, "saving value cache\n");
        writer.write(v->data, sizeof(char), ggml_nbytes(k));
        return true;
    }

    bool KVCacheBuffer::load_state(BinaryFileReader& reader, Logger const& logger) noexcept {
        reader.read(&memory_type);

        logger.log(__func__, "loading key cache\n");
        reader.read(k->data, sizeof(char), ggml_nbytes(k));
        
        logger.log(__func__, "loading value cache\n");
        reader.read(v->data, sizeof(char), ggml_nbytes(k));
        return true;
    }

    // Assumption 1: Layer is not being modified. Therefore, we can skip it
    // Assumption 2: User will only load the state of a correct model
    bool Model::save_state(BinaryFileWriter& writer) const noexcept {
        kv_self.save_state(writer, logger);
        return true;
    }

    bool Model::load_state(BinaryFileReader& reader) noexcept {
        kv_self.load_state(reader, logger);
        return true;
    }


    bool Model::dump_vocab(std::string_view filepath) {
        std::ofstream f{ std::string(filepath) };
        if (!f) return false;

        for(auto i = 0ul; i < vocabulary.id_to_token.size(); ++i) {
            f << i << " : " << vocabulary.id_to_token[i].tok<<'\n';
        }
        f.close();
        return true;
    }


    bool Model::load(std::string_view filepath) {
        using namespace ::fastllama::literals;

        logger.log("Model", "loading model from ", filepath, " - please wait ...\n");
        if (load_parallel){
            logger.log("Model", "parallel loading with block size of ", n_load_parallel_blocks,"\n");
        }
        
        this->is_valid = false;

        // Create model loader

        auto model_loader = ModelLoader(filepath, use_mmap, false, &logger);

        if (model_loader.is_load_failed) {
            return false;
        }

        vocabulary = std::move(model_loader.file_loaders[0].vocab);
        auto n_ctx = params.n_ctx; 
        params = std::move(model_loader.file_loaders[0].hyperparams);
        params.n_ctx = n_ctx;
        file_version = model_loader.file_loaders[0].version;

        std::uint32_t n_ff = ((2*(4*params.n_embd)/3 + params.n_mult - 1)/params.n_mult)*params.n_mult;

        // Get model id
        {
            switch(params.n_layer) {
                case 32: model_id = ModelId::from_str_case_insensitive("7B"); break;
                case 40: model_id = ModelId::from_str_case_insensitive("13B"); break;
                case 60: model_id = ModelId::from_str_case_insensitive("30B"); break;
                case 80: model_id = ModelId::from_str_case_insensitive("65B"); break;
                default: model_id = ModelId::from_str_case_insensitive("7B"); break;
            }
        }

        // Log HyperParams and ModelSize
        {
            logger.log("Model", "n_vocab    = ", params.n_vocab, '\n');
            logger.log("Model", "n_ctx      = ", params.n_ctx, '\n');
            logger.log("Model", "n_embd     = ", params.n_embd, '\n');
            logger.log("Model", "n_mult     = ", params.n_mult, '\n');
            logger.log("Model", "n_head     = ", params.n_head, '\n');
            logger.log("Model", "n_layer    = ", params.n_layer, '\n');
            logger.log("Model", "n_rot      = ", params.n_rot, '\n');
            logger.log("Model", "ftype      = ", static_cast<std::uint32_t>(params.ftype), " (", to_string_view(params.ftype), ")\n");
            logger.log("Model", "n_ff       = ", n_ff, '\n');
            logger.log("Model", "n_parts    = ", model_loader.file_loaders.size(), '\n');
            logger.log("Model", "model_id   = ", model_id, '\n');
        }

        // Initialize cache
        if (!kv_self.init(params, logger)) return false;

        std::size_t ctx_size{};
        std::size_t mmapped_size{};

        if (!model_loader.calc_sizes(&ctx_size, &mmapped_size)) {
            return false;
        }

        char buff[32];

        logger.log("Model", "ggml ctx size = ", humanize_size(buff, ctx_size), '\n');

        // Initialize compute buffers
        {
            model_id.config.mem_required_for_eval += static_cast<std::size_t>(n_batch) * 20_MiB + allocate_extra_mem; // extra space large batch
            buf_compute.resize(model_id.config.mem_required_for_eval);
            if constexpr (use_scratch_buffer) {
                buf_scratch[0].resize(model_id.config.mem_required_for_scratch_buff_0);
                buf_scratch[1].resize(model_id.config.mem_required_for_scratch_buff_1);
            }
        }

        // Log memory requirements 
        {
            std::size_t const scale = kv_self.memory_type == GGML_TYPE_F32 ? 2 : 1;
            auto const mem_required =
                ctx_size +
                mmapped_size +
                (model_id.config.mem_required_for_scratch_buff_0 * static_cast<std::size_t>(use_scratch_buffer)) +
                (model_id.config.mem_required_for_scratch_buff_1 * static_cast<std::size_t>(use_scratch_buffer)) +
                model_id.config.mem_required_for_eval;

            auto const mem_required_state = scale * model_id.config.mem_required_for_kv_self_buff;

            logger.log("Model", "mem required  = ", dyn_humanize_size(mem_required)," (+ ", dyn_humanize_size(mem_required_state) ," per state)\n");
        }
        
        // Set the ggml context
        {
            buffer.resize(ctx_size);

            if (use_mlock) {
                mlock_buffer.init(buffer.data());
                mlock_buffer.grow_to(buffer.size());
            }

            ctx = MemContext(buffer.data(), buffer.size(), model_loader.use_mmap);

            if (!ctx) {
                logger.log_err(__func__, "failed to initialize ggml context\n");
                return false;
            }
        }

        // prepare memory for the weights
        {
            auto const n_embd  = params.n_embd;
            auto const n_layer = params.n_layer;
            auto const n_vocab = params.n_vocab;

            model_loader.mem_ctx = ctx;

            model_loader.mem_ctx = ctx;

            tok_embeddings = model_loader.get_tensor("tok_embeddings.weight", {n_embd, n_vocab});
            norm           = model_loader.get_tensor("norm.weight",           {n_embd});
            output         = model_loader.get_tensor("output.weight",         {n_embd, n_vocab});

            layers.resize(n_layer);

            char buff[256];

            for (auto i = 0; i < static_cast<int>(n_layer); ++i) {
                auto & layer = layers[i];

                layer.attention_norm = model_loader.get_tensor(format_str(buff, "layers.%d.attention_norm.weight", i), {n_embd});

                layer.wq = model_loader.get_tensor(format_str(buff, "layers.%d.attention.wq.weight", i), {n_embd, n_embd});
                layer.wk = model_loader.get_tensor(format_str(buff, "layers.%d.attention.wk.weight", i), {n_embd, n_embd});
                layer.wv = model_loader.get_tensor(format_str(buff, "layers.%d.attention.wv.weight", i), {n_embd, n_embd});
                layer.wo = model_loader.get_tensor(format_str(buff, "layers.%d.attention.wo.weight", i), {n_embd, n_embd});

                layer.ffn_norm = model_loader.get_tensor(format_str(buff, "layers.%d.ffn_norm.weight", i), {n_embd});

                layer.w1 = model_loader.get_tensor(format_str(buff, "layers.%d.feed_forward.w1.weight", i), {n_embd,   n_ff});
                layer.w2 = model_loader.get_tensor(format_str(buff, "layers.%d.feed_forward.w2.weight", i), {  n_ff,   n_embd});
                layer.w3 = model_loader.get_tensor(format_str(buff, "layers.%d.feed_forward.w3.weight", i), {n_embd,   n_ff});
            }

        }

        if (!model_loader.done_getting_tensors()) return false;


        // populate `tensors_by_name`
        tensor_by_name = model_loader.tensors_map.make_tensors_by_name();

        auto time = std::chrono::steady_clock::now();
        if (load_parallel) model_loader.parallel_load_all_data(use_mlock ? &mlock_mmap : nullptr, threads, n_load_parallel_blocks);
        else               model_loader.load_all_data(use_mlock ? &mlock_mmap : nullptr);
        auto time2 = std::chrono::steady_clock::now();
        auto diff = time2 - time;
        logger.log("Model", "time to load all data = ", std::chrono::duration<double, std::milli>(diff).count(), " ms\n");

        if (model_loader.is_load_failed) return false;

        mapping = std::move(model_loader.mmapped_file);
        use_mmap = model_loader.use_mmap;

        this->is_valid = true;
        return true;
    }

    auto Model::eval(
            std::size_t n_past,
            Span<vocab_id> embd_inp,
            std::vector<float>& embd_w,
            std::size_t& mem_per_token
        ) -> bool
    {
        if (!is_valid) {
            logger.log_err(__func__, "model is not valid\n");
            return false;
        };
        auto const N = static_cast<std::int64_t>(embd_inp.size());

        auto const n_embd  = params.n_embd;
        auto const n_layer = params.n_layer;
        auto const n_ctx   = params.n_ctx;
        auto const n_head  = params.n_head;
        auto const n_vocab = params.n_vocab;
        auto const n_rot   = params.n_embd / params.n_head;

        ggml_init_params mem_params {};
        // buf_compute.resize(mem_per_token);
        mem_params.mem_size   = buf_compute.size();
        mem_params.mem_buffer = reinterpret_cast<void*>(buf_compute.data());

        ggml_context * ctx0 = ggml_init(mem_params);
        ggml_cgraph gf{};
        gf.n_threads = (N >= 32 && (ggml_cpu_has_blas() || ggml_cpu_has_cublas()) ? 1 : threads);

        ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        std::copy_n(embd_inp.begin(), N, static_cast<vocab_id*>(embd->data));

        ggml_tensor* inpL = ggml_get_rows(ctx0, tok_embeddings, embd);

        auto const past_size = static_cast<std::int64_t>(n_past);
        
        for (auto il = 0ul; il < layers.size(); ++il) {
            ggml_tensor * inpSA = inpL;

            ggml_tensor * cur;

            use_buf(ctx0, 0);

            // norm
            {
                cur = ggml_rms_norm(ctx0, inpL);

                // cur = attention_norm*cur
                cur = ggml_mul(ctx0,
                            ggml_repeat(ctx0, layers[il].attention_norm, cur),
                            cur);
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor* Qcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layers[il].wq, cur), n_embd/n_head, n_head, N), past_size, n_rot, 0);
                ggml_tensor* Kcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, layers[il].wk, cur), n_embd/n_head, n_head, N), past_size, n_rot, 0);

                // store key and value to memory
                {
                    // compute the transposed [N, n_embd] V matrix
                    ggml_tensor* Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, layers[il].wv, cur), n_embd, N));

                    ggml_tensor* k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + past_size));
                    ggml_tensor* v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + past_size*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
                }

                ggml_tensor* Q =
                    ggml_permute(ctx0,
                            Qcur,
                            0, 2, 1, 3);

                ggml_tensor* K =
                    ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, kv_self.k, (past_size + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
                                n_embd/n_head, n_head, past_size + N),
                            0, 2, 1, 3);

                // K * Q
                ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                ggml_tensor * KQ_scaled =
                    ggml_scale(ctx0,
                            KQ,
                            ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));

                // KQ_masked = mask_past(KQ_scaled)
                ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, past_size);

                // KQ = soft_max(KQ_masked)
                ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

                // split cached V into n_head heads
                ggml_tensor* V =
                    ggml_view_3d(ctx0, kv_self.v,
                            past_size + N, n_embd/n_head, n_head,
                            n_ctx*ggml_element_size(kv_self.v),
                            n_ctx*ggml_element_size(kv_self.v)*n_embd/n_head,
                            il*n_ctx*ggml_element_size(kv_self.v)*n_embd);

                
#if 1
                ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
#else
                // make V contiguous in memory to speed up the matmul, however we waste time on the copy
                // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
                // is there a better way?
                ggml_tensor* V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, past_size + N, n_embd/n_head, n_head));
                ggml_tensor* KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                cur = ggml_cpy(ctx0,
                        KQV_merged,
                        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

                // projection (no bias)
                cur = ggml_mul_mat(ctx0,
                        layers[il].wo,
                        cur);
            }

            use_buf(ctx0, 1);

            ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

            // feed-forward network
            {
                // norm
                {
                    cur = ggml_rms_norm(ctx0, inpFF);

                    // cur = ffn_norm*cur
                    cur = ggml_mul(ctx0,
                            ggml_repeat(ctx0, layers[il].ffn_norm, cur),
                            cur);
                }

                ggml_tensor * tmp = ggml_mul_mat(ctx0,
                        layers[il].w3,
                        cur);

                cur = ggml_mul_mat(ctx0,
                        layers[il].w1,
                        cur);

                // SILU activation
                cur = ggml_silu(ctx0, cur);

                cur = ggml_mul(ctx0, cur, tmp);

                cur = ggml_mul_mat(ctx0,
                        layers[il].w2,
                        cur);
            }

            cur = ggml_add(ctx0, cur, inpFF);

            // input for next layer
            inpL = cur;
        }

        use_buf(ctx0, 0);

        // used at the end to optionally extract the embeddings
        ggml_tensor * embeddings = nullptr;

        // norm
        {

            inpL = ggml_rms_norm(ctx0, inpL);

            // inpL = norm*inpL
            inpL = ggml_mul(ctx0,
                        ggml_repeat(ctx0, norm, inpL),
                        inpL);

            embeddings = inpL;
        }

        // lm_head
        inpL = ggml_mul_mat(ctx0, output, inpL);

        use_buf(ctx0, -1);

        // logits -> probs
        //inpL = ggml_soft_max(ctx0, inpL);

        // run the computation
        ggml_build_forward_expand(&gf, inpL);
        ggml_graph_compute       (ctx0, &gf);

        {
            // return result for just the last token
            auto const embd_w_len = static_cast<std::size_t>(n_vocab) * (should_put_all_logits ? N : 1ul);
            auto const data_offset = static_cast<std::ptrdiff_t>(n_vocab) * (should_put_all_logits ? 0 : N - 1ul);
            embd_w.resize(embd_w_len);
            auto const* data_ptr = static_cast<float*>(ggml_get_data(inpL)) + data_offset;
            std::copy_n(data_ptr, embd_w.size(), embd_w.begin());
        }

        if (embeddings_eval_enable) {
            auto& embeddings_out = this->embeddings;
            embeddings_out.resize(static_cast<std::size_t>(n_embd));
            auto const* ggml_data_ptr = static_cast<float*>(ggml_get_data(embeddings)) + n_embd * (static_cast<int>(N) - 1);
            std::copy_n(ggml_data_ptr, embeddings_out.size(), embeddings_out.begin());
        }

        if (mem_per_token == 0) {
            mem_per_token = ggml_used_mem(ctx0) / N;
        }

        ggml_free(ctx0);

        return true;
    }

    bool quantize(std::string_view in_filepath, std::string_view out_filepath, FType ftype, int threads) {
        using namespace ::fastllama::literals;

        auto logger = Logger();

        ggml_type quantized_type;
        switch (ftype) {
            case FType::MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
            case FType::MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
            case FType::MOSTLY_Q4_2: quantized_type = GGML_TYPE_Q4_2; break;
            case FType::MOSTLY_Q4_3: quantized_type = GGML_TYPE_Q4_3; break;
            default: {
                logger.log_err(__func__, "invalid quantization type ", static_cast<int>(ftype), to_string_view(ftype), '\n');
                return false;
            }
        };

        auto n_threads = std::min(std::max(1, threads), static_cast<int>(std::thread::hardware_concurrency()));
        
        fprintf(stderr, "%s: using %d threads\n", __func__, n_threads);

        fprintf(stderr, "%s: loading model from '%s'", __func__, in_filepath.data());

        auto model_loader = ModelLoader(in_filepath, false, false, &logger);

        if (model_loader.is_load_failed) {
            logger.log_err(__func__, "failed to load model from '", in_filepath, "'\n");
            return false;
        }

        auto file_saver = FileSaver(out_filepath, &model_loader.file_loaders[0], ftype, &logger);


        // load -> transform -> save weights
        {
            std::size_t total_size_org = 0;
            std::size_t total_size_new = 0;

            std::vector<std::thread> workers;
            std::mutex mtx;

            std::vector<std::int64_t> hist_all(1 << 4, 0);


            char str_buff[256];

            for(auto i = 0ul; i < model_loader.tensors_map.tensors.size(); ++i) {
                auto& tensor = model_loader.tensors_map.tensors[i];

                UninitializedBuffer buff(tensor.size);
                tensor.data = buff.data();

                model_loader.load_data_for(tensor);

                fprintf(stderr, "%s: [%4zu/%4zu] %36s - %16s, type = %6s, ", __func__, i, model_loader.tensors_map.tensors.size(),
                    tensor.name.c_str(), format_tensor_shape(tensor.extents).c_str(), ggml_type_name(tensor.type));

                bool quantize = (tensor.extents.size() == 2);

                static std::string_view const suffixes[] = {
                    "weight",
                    "weight.lora",
                    "weight.loraA",
                    "weight.loraB",
                };

                for(auto const& suffix : suffixes) {
                    if (tensor.name.rfind(suffix) == (tensor.name.size() - suffix.size())) {
                        quantize &= true;
                        break;
                    }
                }

                enum ggml_type new_type;
                void * new_data;
                size_t new_size;
                UninitializedBuffer work;

                if (!quantize) {
                    new_type = tensor.type;
                    new_data = tensor.data;
                    new_size = tensor.size;
                } else {
                    new_type = quantized_type;
                    float * f32_data;
                    std::size_t n_elements = tensor.extents[0] * tensor.extents[1];
                    UninitializedBuffer f32_conv_buf;

                    if (tensor.type == GGML_TYPE_F32) {
                        f32_data = reinterpret_cast<float*>(tensor.data);
                    } else if (tensor.type == GGML_TYPE_F16) {
                        f32_conv_buf.resize(n_elements * sizeof(float));
                        f32_data = reinterpret_cast<float*>(f32_conv_buf.data());
                        auto f16_data = reinterpret_cast<ggml_fp16_t*>(tensor.data);
                        for (std::size_t i = 0ul; i < n_elements; ++i) f32_data[i] = ggml_fp16_to_fp32(f16_data[i]);
                    } else {
                        logger.log_err(__func__, "unsupported for integer quantization ", ggml_type_name(tensor.type), '\n');
                        return false;
                    }
                    
                    fprintf(stderr, "%s: quantizing...\n", __func__);
                    fflush(stdout);

                    work.resize(n_elements * 4);
                    new_data = work.data();
                    std::vector<int64_t> hist_cur(1 << 4, 0);

                    int chunk_size = 32 * 512;

                    int const nchunk = (n_elements + chunk_size - 1)/chunk_size;
                    int const nthread_use = n_threads > 1 ? std::max(1, std::min(n_threads, nchunk)) : 1;

                    if (nthread_use < 2) {
                        new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, n_elements, hist_cur.data());
                    } {
                        size_t counter = 0;
                        new_size = 0;
                        auto compute = [&mtx, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, n_elements, chunk_size] () {
                            std::vector<int64_t> local_hist;
                            size_t local_size = 0;
                            while (true) {
                                std::unique_lock<std::mutex> lock(mtx);
                                size_t first = counter; counter += chunk_size;
                                if (first >= n_elements) {
                                    if (!local_hist.empty()) {
                                        for (int j=0; j<int(local_hist.size()); ++j) hist_cur[j] += local_hist[j];
                                        new_size += local_size;
                                    }
                                    break;
                                }
                                lock.unlock();
                                size_t last = std::min(n_elements, first + chunk_size);
                                if (local_hist.empty()) local_hist.resize(hist_cur.size(), 0);
                                local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                            }
                        };
                        
                        if (int(workers.size()) < nthread_use - 1) workers.resize(nthread_use - 1);
                        
                        for (int it = 0; it < nthread_use - 1; ++it) workers[it] = std::thread(compute);
                        
                        compute();
                        
                        for (int it = 0; it < nthread_use - 1; ++it) workers[it].join();
                    }

                    printf("size = %8.2f MB -> %8.2f MB | hist: ", tensor.size/1024.0/1024.0, new_size/1024.0/1024.0);
                    for (size_t i = 0; i < hist_cur.size(); i++) {
                        hist_all[i] += hist_cur[i];
                    }

                    for (size_t i = 0; i < hist_cur.size(); i++) {
                        printf("%5.3f ", hist_cur[i] / float(n_elements));
                    }
                    printf("\n");
                }

                total_size_org += tensor.size;
                total_size_new += new_size;
                file_saver.write_tensor(tensor, new_type, new_data, new_size);
            }
            fprintf(stderr, "\n%s: model size  = %8.2f MB\n",__func__, static_cast<float>(total_size_org / 1.0_MiB));
            fprintf(stderr, "%s: quant size  = %8.2f MB\n",__func__, static_cast<float>(total_size_new / 1.0_MiB));

            {
                int64_t sum_all = 0;
                for (size_t i = 0; i < hist_all.size(); i++) {
                    sum_all += hist_all[i];
                }

                printf("%s: hist: ", __func__);
                for (size_t i = 0; i < hist_all.size(); i++) {
                    printf("%5.3f ", hist_all[i] / float(sum_all));
                }
                printf("\n");
            }
        }

        return true;
    }

    inline static bool reset_tensor_data_ptrs(Model& model) {
        auto const& logger = model.logger;

        for(auto& [name, data] : model.org_tensor_data_ptr_for_mmap) {
            auto tensor_it = model.tensor_by_name.find(name);
            if (tensor_it == model.tensor_by_name.end()) {
                logger.log_err(__func__, "tensor '", name, "' not found\n");
                return false;
            }
            tensor_it->second->data = data;
        }

        return true;
    }

    template<typename Fn>
    inline static bool attach_or_detach_lora_helper(std::string_view filepath, Model& model, Fn&& adapter_fn, const char* func_name, bool is_detach = false) {
        using namespace literals;
        using tensor_map_t = std::unordered_map<std::string, ggml_tensor*>;
        static_assert(std::is_invocable_r_v<
            ggml_tensor*,
            Fn,
            MemContext&,
            ggml_tensor*,
            ggml_tensor*
        >);

        auto const& logger = model.logger;

        auto const set_lora_file_path = [&] {
            if (is_detach) model.attached_lora_path = "";
            else model.attached_lora_path = filepath;
        };

        if (is_detach && model.use_mmap) {
            model.buffer_lora_head = 0;
            logger.log(func_name, "freeing ", dyn_humanize_size(model.buffer_lora_for_mmap.size()), '\n');
            if (!reset_tensor_data_ptrs(model)) {
                model.buffer_lora_for_mmap.free();
                set_lora_file_path();
                return false;
            }
            model.buffer_lora_for_mmap.free();
            set_lora_file_path();
            return true;
        }


        auto model_loader = ModelLoader(filepath, false, false, &logger);

        if (model_loader.is_load_failed) {
            return false;
        }

        auto const lora_params = model_loader.file_loaders[0].lora_adapter_params;
        
        // Log the lora parameters
        if (!is_detach) {
            char buff[16];
            logger.log(func_name, "lora_params: \n");
            logger.log(func_name, "   Use cached matrix= ", lora_params.use_cache_matrix ? "Yes" : "No", '\n');
            logger.log(func_name, "   Alpha            = ", lora_params.alpha, '\n');
            logger.log(func_name, "   Rank             = ", lora_params.r, '\n');
            logger.log(func_name, "   Scale            = ", format_str(buff, "%.2f", lora_params.get_scale()), '\n');

        }

        bool use_cache = lora_params.use_cache_matrix;

        float scale{lora_params.get_scale()};
        
        UninitializedBuffer buffer(1_GiB);

        struct LoraTensor{
            ggml_tensor* a{nullptr};
            ggml_tensor* b{nullptr};
        };

        std::unordered_map<std::string, LoraTensor> lora_tensors;

        auto data_loaded = std::size_t{};
        auto total_size = model_loader.total_size_needed_for_the_tensors();

        bool warned{false};

        auto const get_base_name = [&](std::string_view name) -> std::optional<std::string> {
            static std::string_view lora_suffixes[] = { ".lora" };

            auto has_lora_suffix = false;
            auto pos = name.rfind(lora_suffixes[0]);
            // cached matrix has no type marker
            auto suffix_start_pos = name.size() - lora_suffixes[0].size() - (use_cache ? 0 : 1);
            has_lora_suffix = pos == suffix_start_pos;

            if (!has_lora_suffix) {
                logger.log_err(func_name, "'", name, "' is not a lora tensor", "\n");
                return {};
            }

            return std::string(name.substr(0, pos));
        };

        if (model.use_mmap) {

            auto possible_lora_ctx_size_for_mmap = std::size_t{};
            
            // possible context size for mmap
            for(auto& tensors: model_loader.tensors_map.tensors) {
                auto const base_name_maybe = get_base_name(tensors.name);
                if (!base_name_maybe) continue;
                auto const& base_name = *base_name_maybe;
                auto it = model.tensor_by_name.find(base_name);
                if (it == model.tensor_by_name.end()) continue;
                possible_lora_ctx_size_for_mmap += ggml_nbytes(it->second) + sizeof(ggml_tensor) + GGML_OBJECT_SIZE;
            }
            possible_lora_ctx_size_for_mmap = possible_lora_ctx_size_for_mmap;
            model.buffer_lora_for_mmap = UninitializedBuffer(possible_lora_ctx_size_for_mmap);
            logger.log(func_name, "Extra memory used = ", dyn_humanize_size(possible_lora_ctx_size_for_mmap), '\n');
        }

        for(auto& lora_tl : model_loader.tensors_map.tensors) {
            if (!model_loader.mem_ctx) {
                model_loader.mem_ctx = MemContext(buffer);
            }
            
            auto const name = lora_tl.name;
            auto const base_name_maybe = get_base_name(name);
            if (!base_name_maybe) return false;
            auto const base_name = *base_name_maybe;

            if (model.tensor_by_name.count(base_name) == 0) {
                logger.log_err(func_name, "unknown tensor '", base_name, "' in lora adapter\n");
                return false;
            }

            if (!use_cache && lora_tl.type != GGML_TYPE_F32) {
                logger.log_err(func_name, "currently, we support fp16 for uncached matrix.\n");
                return false;
            }
            
            // Construct the lora tensor
            auto* current_lora_tensor = model_loader.get_tensor_for(lora_tl);
            
            // Loads the tensor into the memory
            model_loader.load_lora_adapter_for(lora_tl);
            
            auto& temp_lora_tensor = lora_tensors[base_name];
            temp_lora_tensor.a = name.back() == 'A' ? current_lora_tensor : temp_lora_tensor.a;
            temp_lora_tensor.b = name.back() == 'B' ? current_lora_tensor : temp_lora_tensor.b;

            // Lora A tensors are scaled during the generation of the lora adapter file
            // so we need to scale them during runtime
            auto loraAs_tensor = use_cache ? current_lora_tensor : temp_lora_tensor.a;
            auto loraB_tensor = use_cache ? current_lora_tensor : temp_lora_tensor.b;

            auto has_loraA = use_cache ? true : loraAs_tensor != nullptr;
            auto has_loraB = use_cache ? true : loraB_tensor != nullptr;

            if (!warned) {
                auto* base_tensor = model.tensor_by_name[base_name];
                if (ggml_is_quantized(base_tensor->type)) {
                    logger.log_warn(func_name, "using a lora adapter with a quantized model may result in poor quality, use a f16 or f32 base model\n");
                    warned = true;
                }
            }
            

            if (has_loraA && has_loraB) {
                auto* base_tensor = model.tensor_by_name[base_name];

                if (use_cache) {
                    if (base_tensor->ne[0] != current_lora_tensor->ne[0] || base_tensor->ne[1] != current_lora_tensor->ne[1]) {
                        logger.log_err(func_name, "incompatible tensor dimensions (", base_tensor->ne[0], " and ", current_lora_tensor->ne[1], ")", " are you sure that this adapter is for this model?\n");
                        return false;
                    }
                } else {
                    if (base_tensor->ne[0] != loraAs_tensor->ne[1] || base_tensor->ne[1] != loraB_tensor->ne[1]) {
                        logger.log_err(func_name, "incompatible tensor dimensions (", base_tensor->ne[0], " and ", loraAs_tensor->ne[1], ")", " are you sure that this adapter is for this model?\n");
                        return false;
                    }
                }

                ggml_tensor* dest_tensor = base_tensor;
                if (model.use_mmap) {
                    auto tensor_data = reinterpret_cast<void*>(model.buffer_lora_for_mmap.data() + model.buffer_lora_head);
                    model.buffer_lora_head += ggml_nbytes(base_tensor);
                    std::memcpy(tensor_data, base_tensor->data, ggml_nbytes(base_tensor));
                    model.org_tensor_data_ptr_for_mmap[base_name] = base_tensor->data;
                    base_tensor->data = tensor_data;
                }

                ggml_tensor* BAs = (use_cache ? current_lora_tensor : ggml_mul_mat(model_loader.mem_ctx, loraAs_tensor, loraB_tensor));

                auto* r = adapter_fn(model_loader.mem_ctx, dest_tensor, BAs);

                ggml_cgraph gf = ggml_build_forward(r);
                gf.n_threads = model.threads;
                ggml_graph_compute(model_loader.mem_ctx, &gf);

                model_loader.mem_ctx.free();
                
            }

            data_loaded += ggml_nelements(current_lora_tensor);
            logger.progress(is_detach ? ProgressTag::DetachLoraAdapter : ProgressTag::AttachLoraAdapter, data_loaded, total_size);
            
        }

        if (!model_loader.done_getting_tensors()) {
            logger.log_err(func_name, "failed to load all tensors\n");
            return false;
        }

        fprintf(stderr, "\n");

        set_lora_file_path();
        return true;
    }

    bool Model::attach_lora(std::string_view filepath) {
        if (!attached_lora_path.empty()) {
            logger.log_err(__func__, "already attached LoRa model from '", attached_lora_path, "'. Detach it first or reload the model.\n");
            return false;
        }
        logger.log(__func__, "attaching LoRa model from '", filepath, "'. Please wait ...\n");
        return attach_or_detach_lora_helper(
            filepath,
            *this,
            [](
                MemContext& ctx,
                ggml_tensor* base_t,
                ggml_tensor* lora_tensor
            ) {
            // W = W + B*(A * scale)
            // W = W + BA * scale
            return ggml_add_inplace(ctx, base_t, lora_tensor);
        }, __func__, false);
    }

    bool Model::detach_lora() {
        if (attached_lora_path.empty()) {
            logger.log_err(__func__, "no LoRa model attached.\n");
            return false;
        }

        logger.log(__func__, "detaching LoRa model from '", attached_lora_path, "'. Please wait ...\n");
        
        return attach_or_detach_lora_helper(
            attached_lora_path,
            *this,
            [](
                MemContext& ctx,
                ggml_tensor* base_t,
                ggml_tensor* lora_tensor
            ) {
            ggml_tensor* factor = ggml_new_f32(ctx, -1.f);
            ggml_tensor* inv_add = ggml_scale(ctx, lora_tensor, factor);

            // W = W - (B(A * scale))
            // W = W - BA * scale
            return ggml_add_inplace(ctx, base_t, inv_add);
        }, __func__, true);
    }

    bool Model::reset() noexcept {
        embeddings.clear();
        return true;
    }

} // namespace fastllama
