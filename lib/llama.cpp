#include "llama.hpp"
#include "file_reader.hpp"
#include "file_pipe.hpp"
#include <cassert>
#include "macro.hpp"
#include <fstream>
#include <numeric>
#include <functional>
#include "span.hpp"
#include <cstring>
#include <cmath>
#include "file_loader.hpp"

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

        char buff[20] = {0};

        logger.log("KVCacheBuffer::init", "kv self size  = ", format_str(buff, "%7.2f", static_cast<float>(total_kv_size / 1.0_MiB)), " MB\n");
        
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
        
        this->is_valid = false;

        // Create model loader

        auto model_loader = ModelLoader(filepath, use_mmap, false, logger);

        if (model_loader.is_load_failed) {
            return false;
        }

        vocabulary = std::move(model_loader.file_loaders[0].vocab);
        params = std::move(model_loader.file_loaders[0].hyperparams);
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
            logger.log("Model", "ftype      = ", static_cast<std::uint32_t>(params.ftype), " (", params.ftype, ")\n");
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

        logger.log("Model", "ggml ctx size = ", format_str(buff, "%6.2f KB", static_cast<float>(ctx_size / 1.0_KiB)), '\n');

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

            char buff[256];
            auto str = format_str(buff, "mem required  = %7.2f MB (+ %7.2f MB per state)\n", static_cast<float>(mem_required / 1.0_MiB), static_cast<float>(mem_required_state / 1.0_MiB));
            logger.log("Model", str);
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

        if (!model_loader.done_getting_tensors()) {
            return false;
        }


        // populate `tensors_by_name`
        tensor_by_name = model_loader.tensors_map.make_tensors_by_name();

        model_loader.load_all_data(use_mlock ? &mlock_mmap : nullptr);

        if (!model_loader.is_load_failed) {
            return false;
        }

        mapping = std::move(model_loader.mmapped_file);

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
        gf.n_threads = (N >= 32 && ggml_cpu_has_blas() ? 1 : threads);

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
                logger.log_err(__func__, "invalid quantization type ", static_cast<int>(ftype), ftype);
                return false;
            }
        };

        auto n_threads = std::min(std::max(1, threads), static_cast<int>(std::thread::hardware_concurrency()));
        
        logger.log(__func__, "using ", n_threads, " threads");

        logger.log(__func__, "loading model from '", in_filepath, "'");

        auto model_loader = ModelLoader(in_filepath, false, false, logger);

        auto file_saver = FileSaver(out_filepath, logger);

        if (magic != MagicKind::GGLA) {
            auto params = HyperParams{};
            pipe.read_and_write(&params.n_vocab, 1, param_print_fn("n_vocab"));
            pipe.read_and_write(&params.n_embd, 1, param_print_fn("n_embd"));
            pipe.read_and_write(&params.n_mult, 1, param_print_fn("n_mult"));
            pipe.read_and_write(&params.n_head, 1, param_print_fn("n_head"));
            pipe.read_and_write(&params.n_layer, 1, param_print_fn("n_layer"));
            pipe.read_and_write(&params.n_rot, 1, param_print_fn("n_rot"));
            pipe.read_and_write(&params.f16, 1, [parent_fn_name, itype](auto* v) {
                *v = itype;
                printf("%.*s: %s = %d\n", static_cast<int>(parent_fn_name.size()), parent_fn_name.data(), "f16", *v);
                return v;
            });

            printf("%s: vocabulary processing started\n", __func__);
            // load and save vocab
            {
                std::string word(64, ' ');

                for(auto i = 0l; i < params.n_vocab; ++i) {
                    std::uint32_t len;

                    pipe.read_and_write(&len);
                    word.resize(len);
                    pipe.read_and_write(word.data(), len);

                    float score{};
                    pipe.read_and_write(&score);
                }
            }

            printf("%s: vocabulary processing completed\n", __func__);
        }


        // load -> transform -> save weights
        {
            std::size_t total_size_org = 0;
            std::size_t total_size_new = 0;

            std::vector<float> work;

            std::vector<std::uint8_t>   data_u8;
            std::vector<ggml_fp16_t>    data_f16;
            std::vector<float>          data_f32;

            std::vector<std::int64_t> hist_all(1 << 4, 0);

            std::string name(64, ' ');

            while(!pipe.get_reader().eof()) {
                std::int32_t n_dims;
                std::int32_t length;
                std::int32_t ftype;

                pipe.read_and_write(&n_dims);
                pipe.read_and_write(&length);
                pipe.get_reader().read(&ftype);

                if (pipe.get_reader().eof()) break;

                std::size_t total_number_of_elements{1};
                std::int32_t extents[2] = { 1, 1 };

                for(std::int32_t i {}; i < n_dims; ++i) {
                    pipe.get_reader().read(extents + i);
                    total_number_of_elements *= static_cast<std::size_t>(extents[i]);
                }
                
                auto const str_len = static_cast<std::size_t>(length);
                name.resize(str_len);
                pipe.get_reader().read(name.data(), str_len);

                {
                    // ensure tensor data is aligned
                    auto maybe_offset = pipe.get_reader().tell();
                    if (!maybe_offset) return false;
                    auto offset = *maybe_offset;
                    offset = (offset + static_cast<std::size_t>(31)) & static_cast<std::size_t>(-32);
                    pipe.get_reader().seek(offset);
                }

                {
                    static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                    printf("%48s - [%5d, %5d], type = %6s ", name.data(), extents[0], extents[1], ftype_str[ftype]);
                }

                std::string_view temp_w_str = "weight";
                auto find_pos = name.rfind(temp_w_str);
                bool quantize = (n_dims == 2) && (find_pos != std::string::npos);
                
                // printf("Quantize? %s, Size: %zu - %zu == %zu\n", quantize ? "Yes" : "No", name.size(), find_pos, temp_w_str.size());

                if (quantize) {
                    if (ftype != 0 && ftype != 1) {
                        fprintf(stderr, "%s: unsupported ftype %d for integer quantization\n", __func__, ftype);
                        return false;
                    }

                    if (ftype == 1) {
                        data_f16.resize(total_number_of_elements);
                        pipe.get_reader().read(data_f16.data(), total_number_of_elements);

                        data_f32.resize(total_number_of_elements);
                        auto const n = static_cast<std::ptrdiff_t>(total_number_of_elements);
                        #pragma omp parallel for if (n > 512)
                        for (auto i = 0l; i < n; ++i) {
                            data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                        }
                    } else {
                        data_f32.resize(total_number_of_elements);
                        pipe.get_reader().read(data_f32.data(), total_number_of_elements);
                    }

                    ftype = itype;
                } else {
                    auto const bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                    data_u8.resize(total_number_of_elements * bpe);
                    pipe.get_reader().read(reinterpret_cast<void*>(data_u8.data()), bpe, total_number_of_elements);
                }

                pipe.get_writer().write(&ftype);

                for(auto i = 0ul; i < static_cast<std::size_t>(n_dims); ++i) {
                    pipe.get_writer().write(extents + i);
                }

                pipe.get_writer().write(name.data(), str_len);

                {
                    // ensure tensor data is aligned
                    auto maybe_offset = pipe.get_writer().tell();
                    if (!maybe_offset) return false;
                    auto offset = *maybe_offset;
                    offset = (offset + static_cast<std::size_t>(31)) & static_cast<std::size_t>(-32);
                    pipe.get_writer().seek(offset);
                }

                if (quantize) {
                    printf("quantizing .. ");
                    work.resize(total_number_of_elements); // for quantization

                    size_t cur_size = 0;
                    std::vector<int64_t> hist_cur(1 << 4, 0);

                    switch (type) {
                        case GGML_TYPE_Q4_0:
                            {
                                cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), static_cast<int>(total_number_of_elements), extents[0], hist_cur.data());
                            } break;
                        case GGML_TYPE_Q4_1:
                            {
                                cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), static_cast<int>(total_number_of_elements), extents[0], hist_cur.data());
                            } break;
                        default:
                            {
                                fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                                return false;
                            }
                    }

                    pipe.get_writer().write(reinterpret_cast<char*>(work.data()), cur_size);

                    total_size_new += cur_size;

                    printf("size = %8.2Lf MB -> %8.2Lf MB | hist: ", total_number_of_elements * sizeof(float)/1.0_MiB, cur_size/1.0_MiB);
                    std::copy(hist_cur.begin(), hist_cur.end(), hist_all.begin());

                    for (auto i = 0ul; i < hist_cur.size(); ++i) {
                        printf("%5.3f ", static_cast<double>(hist_cur[i]) / static_cast<double>(total_number_of_elements));
                    }
                    printf("\n");

                } else {
                    printf("size = %8.3Lf MB\n", data_u8.size()/1.0_MiB);
                    pipe.get_writer().write(data_u8.data(), data_u8.size());

                    total_size_new += data_u8.size();
                }

                total_size_org += total_number_of_elements * sizeof(float);
            }

            printf("%s: model size  = %8.2Lf MB\n", __func__, total_size_org / 1.0_MiB);
            printf("%s: quant size  = %8.2Lf MB\n", __func__, total_size_new / 1.0_MiB);

            {
                double sum_all = std::accumulate(hist_all.begin(), hist_all.end(), double{}, std::plus<>{});

                printf("%s: hist: ", __func__);
                for (auto i = 0ul; i < hist_all.size(); ++i) {
                    printf("%5.3f ", static_cast<double>(hist_all[i]) / sum_all);
                }
                printf("\n");
            }

        }

        pipe.close();

        return true;
    }

    template<typename Fn>
    inline static bool attach_or_detach_lora_helper(std::string_view filepath, Model& model, Fn&& adapter_fn, const char* func_name, bool is_detach = false) {
        using tensor_map_t = std::unordered_map<std::string, ggml_tensor*>;
        static_assert(std::is_invocable_r_v<
            ggml_tensor*,
            Fn,
            ggml_context*,
            ggml_tensor*,
            tensor_map_t&,
            std::string const&,
            std::string const&,
            std::string const&,
            float,
            bool
        >);

        using namespace literals;
        auto const& logger = model.logger;

        auto reader = BinaryFileReader(filepath);
        if (!reader) {
            logger.log_err(func_name, "failed to open file ", filepath, "\n");
            return false;
        }

        if (!verify_magic_number(reader)) {
            logger.log_err(func_name, "bad file magic\n");
            return false;
        }

        if (!verify_file_version(reader)) {
            logger.log_err(func_name, "unsupported file version\n");
            return false;
        }

        bool use_cache = false;
        reader.read(&use_cache);

        std::uint32_t r{1};
        std::uint32_t alpha{1};
        float scale{1.0f};

        if (use_cache) {
            logger.log(func_name, "cached adapter found\n");
        } else {
            logger.log(func_name, "uncached adapter found\n");
            reader.read(&r);
            reader.read(&alpha);
            scale = static_cast<float>(alpha) / static_cast<float>(r);
            char format_buffer[64];
            logger.log(func_name, "r = ", r, ", alpha = ", alpha, format_str(format_buffer, ", scale = %.2f", scale), "\n");
        }

        ggml_context* lora_ctx{nullptr};

        std::vector<unsigned char> buff(1_GiB);
        ggml_init_params params = {};
        params.mem_buffer = buff.data();
        params.mem_size = buff.size();
        params.no_alloc = false;

        lora_ctx = ggml_init(params);

        tensor_map_t lora_tensors;
        tensor_map_t model_tensors = model.tensors.make_tensors_by_name();

        auto n_tensors = std::size_t{};

        bool warned{false};

        while(!reader.eof()) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            reader.read(&n_dims);
            reader.read(&length);
            reader.read(&ftype);

            if (reader.eof()) break;

            int32_t ne[2] = { 1, 1 };
            for(auto i = 0; i < n_dims; ++i) {
                reader.read(ne + i);
            }

            std::string name(length, '\0');
            reader.read(name.data(), length);

            std::string_view lora_suffix = ".lora";
            auto lora_pos = name.rfind(lora_suffix);
            if (lora_pos == std::string::npos) {
                logger.log_err(func_name, "'", name, "' is not a lora tensor", "\n");
                return false;
            }

            auto base_name = name.substr(0, lora_pos);

            if (!model.tensors.contains(base_name)) {
                logger.log_err(func_name, "unknown tensor '", base_name, "' in lora adapter\n");
                return false;
            }

            ggml_type wtype;
            switch (ftype) {
                case 0: wtype = GGML_TYPE_F32;  break;
                case 1: wtype = GGML_TYPE_F16;  break;
                case 2: wtype = GGML_TYPE_Q4_0; break;
                case 3: wtype = GGML_TYPE_Q4_1; break;
                default:{
                    logger.log_err(func_name, "unsupported lora tensor type ", ftype, "\n");
                    return false;
                }
            }

            if (!use_cache && wtype != GGML_TYPE_F32) {
                logger.log_err(func_name, "currently, we support fp32 for uncached matrix.\n");
                return false;
            }

            ggml_tensor* lora_tensor;

            if (n_dims == 2) {
                lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
            } else {
                logger.log_err(func_name, "unsupported tensor dimension ", n_dims, "\n");
                return false;
            }
            
            {
                auto maybe_offset = reader.tell();
                if (!maybe_offset) {
                    logger.log_err(func_name, "failed to get file offset\n");
                    return false;
                }
                auto offset = *maybe_offset;
                size_t tensor_data_size = ggml_nbytes(lora_tensor);
                offset = (offset + 31) & -32;
                reader.seek(offset);
            }

            reader.read(lora_tensor->data, sizeof(char), ggml_nbytes(lora_tensor));

            // BA matrix with scaled values in case of cached adapter
            lora_tensors[name] = lora_tensor;

            auto loraA_str = base_name + ".loraA";
            auto loraB_str = base_name + ".loraB";

            auto has_loraA = lora_tensors.find(loraA_str) != lora_tensors.end();
            auto has_loraB = lora_tensors.find(loraB_str) != lora_tensors.end();

            if (use_cache || (has_loraA && has_loraB)) {
                ggml_tensor* base_t = model_tensors[base_name];
                if (!warned) {
                    if (base_t->type == GGML_TYPE_Q4_0 || base_t->type == GGML_TYPE_Q4_1) {
                        logger.log_warn(func_name, "using a lora adapter with a quantized model may result in poor quality, use a f16 or f32 base model\n");
                        warned = true;
                    }
                }
                
                if (use_cache) {
                    if (base_t->ne[0] != lora_tensor->ne[0] || base_t->ne[1] != lora_tensor->ne[1]) {
                        logger.log_err(func_name, "incompatible tensor dimensions (", base_t->ne[0], " and ", lora_tensor->ne[1], ")", " are you sure that this adapter is for this model?\n");
                        return false;
                    }
                } else {
                    auto* loraA_t = lora_tensors[loraA_str];
                    auto* loraB_t = lora_tensors[loraB_str];
                    if (base_t->ne[0] != loraA_t->ne[1] || base_t->ne[1] != loraB_t->ne[1]) {
                        logger.log_err(func_name, "incompatible tensor dimensions (", base_t->ne[0], " and ", loraA_t->ne[1], ")", " are you sure that this adapter is for this model?\n");
                        return false;
                    }
                }

                auto* r = adapter_fn(lora_ctx, base_t, lora_tensors, name, loraA_str, loraB_str, scale, use_cache);

                ggml_cgraph gf = ggml_build_forward(r);
                gf.n_threads = model.threads;
                ggml_graph_compute(lora_ctx, &gf);
                
                // we won't need these tensors again, reset the context to save memory
                ggml_free(lora_ctx);
                lora_ctx = ggml_init(params);
                lora_tensors.clear();

                n_tensors++;
                if (n_tensors % 4 == 0) fprintf(stderr, ".");
            }



        }
        ggml_free(lora_ctx);
        fprintf(stderr, "\n");

        if (is_detach) model.attached_lora_path = "";
        else model.attached_lora_path = filepath;
        return true;
    }

    bool Model::attach_lora(std::string_view filepath) {
        using namespace literals;
        if (!attached_lora_path.empty()) {
            logger.log_err(__func__, "already attached LoRa model from ", attached_lora_path, ". Detach it first or reload the model.\n");
            return false;
        }
        logger.log(__func__, "attaching LoRa model from ", filepath, ". Please wait ...\n");
        return attach_or_detach_lora_helper(
            filepath,
            *this,
            [](
                ggml_context* ctx,
                ggml_tensor* base_t,
                std::unordered_map<std::string, ggml_tensor*>& lora_tensors,
                std::string const& lora_name,
                std::string const& loraA_str,
                std::string const& loraB_str,
                float scale,
                bool use_cache
            ) {
            ggml_tensor* lora_t = nullptr;
            if (use_cache) {
                lora_t = lora_tensors[lora_name];
            } else {
                auto loraA = lora_tensors[loraA_str];
                auto loraB = lora_tensors[loraB_str];
                // BA matrix
                lora_t = ggml_mul_mat(ctx, loraA, loraB);
                if (scale != 1.f) {
                    ggml_tensor* factor = ggml_new_f32(ctx, scale);
                    lora_t = ggml_scale(ctx, lora_t, factor);
                }
            }
            
            // W = W + BA * scale
            return ggml_add_inplace(ctx, base_t, lora_t);
        }, __func__, false);
    }

    bool Model::detach_lora() {
        if (attached_lora_path.empty()) {
            logger.log_err(__func__, "no LoRa model attached.\n");
            return false;
        }

        logger.log(__func__, "detaching LoRa model from ", attached_lora_path, ". Please wait ...\n");
        
        return attach_or_detach_lora_helper(
            attached_lora_path,
            *this,
            [](
                ggml_context* ctx,
                ggml_tensor* base_t,
                std::unordered_map<std::string, ggml_tensor*>& lora_tensors,
                std::string const& lora_name,
                std::string const& loraA_str,
                std::string const& loraB_str,
                float scale,
                bool use_cache
            ) {
            ggml_tensor* lora_t = nullptr;
            ggml_tensor* factor = ggml_new_f32(ctx, -1.f * scale);
            if (use_cache) {
                lora_t = lora_tensors[lora_name];
            } else {
                auto loraA = lora_tensors[loraA_str];
                auto loraB = lora_tensors[loraB_str];
                // BA matrix
                lora_t = ggml_mul_mat(ctx, loraA, loraB);
            }

            ggml_tensor* inv_add = ggml_scale(ctx, lora_t, factor);
            // W = W - BA * scale
            return ggml_add_inplace(ctx, base_t, inv_add);
        }, __func__, true);

        // Why is the sun flat?

        return true;
    }

    bool Model::reset() noexcept {
        embeddings.clear();
        return true;
    }

} // namespace fastllama
