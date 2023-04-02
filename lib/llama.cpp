#include "llama.hpp"
#include "file_reader.hpp"
#include <cassert>
#include "macro.hpp"
#include <thread>

namespace fastllama {

    FASTLLAMA_ALWAYS_INLINE static constexpr auto verify_magic_number(BinaryFileReader& reader) noexcept -> bool {
        std::uint32_t magic{};
        reader.read(&magic);
        return magic == magic_number_v;
    }

    FASTLLAMA_ALWAYS_INLINE static constexpr auto load_hyperparams(BinaryFileReader& reader, HyperParams& params) {
        reader.read(&params.n_vocab);
        reader.read(&params.n_embd);
        reader.read(&params.n_mult);
        reader.read(&params.n_head);
        reader.read(&params.n_layer);
        reader.read(&params.n_rot);
        reader.read(&params.f16);
    }

    FASTLLAMA_ALWAYS_INLINE static auto load_vocab(BinaryFileReader& reader, Vocab& vocab, std::size_t size) {
        vocab.id_to_token.resize(size);
        std::string word(64, ' ');

        for (auto i = 0ul; i < size; ++i) {
            std::uint32_t len;
            reader.read(&len);

            word.reserve(len);
            reader.read(word.data(), len);

            float score{};

            vocab.token_to_id[word] = i;
            auto& temp = vocab.id_to_token[i];
            temp.score = score;
            temp.tok = word;
            word.clear();
        }
    }

    FASTLLAMA_ALWAYS_INLINE static auto prepare_memory_for_weight(Model& model, ggml_type wtype, ggml_type wtype2, int n_ff) {
        auto const& params = model.params;
        auto const n_embd = params.n_embd;
        auto const n_layer = params.n_layer;
        auto const n_ctx = params.n_ctx;
        auto const n_vocab = params.n_vocab;
        auto& ctx = model.ctx;

        model.layers.resize(n_layer);
        model.tok_embeddings    = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.norm              = ggml_new_tensor_1d(ctx, wtype2, n_embd);
        model.output            = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;

        model.tensors["norm.weight"]   = model.norm;
        model.tensors["output.weight"] = model.output;

        char temp_buff[128] = {0};

        for(auto i = 0ul; i < n_layer; ++i) {
            auto& layer = model.layers[i];
            layer.attention_norm = ggml_new_tensor_1d(ctx, wtype2, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, wtype2, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);

            auto str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention_norm.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.attention_norm;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wq.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wq;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wk.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wq;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wv.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wv;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wo.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wo;
            
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.ffn_norm.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.ffn_norm;
            
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.feed_forward.w1.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.w1;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.feed_forward.w2.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.w2;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.feed_forward.w3.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.w3;
        }
    }

    FASTLLAMA_ALWAYS_INLINE static auto prepare_memory_for_key_value_memory(Model& model) -> std::size_t {
        auto const& params = model.params;

        auto const n_embd  = params.n_embd;
        auto const n_layer = params.n_layer;
        auto const n_ctx   = params.n_ctx;

        auto const n_mem      = n_layer*n_ctx;
        auto const n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_elements);

        std::size_t const memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);
        return memory_size;
    }

    FASTLLAMA_ALWAYS_INLINE static auto load_model_weights(BinaryFileReader& reader, Model& model, std::size_t part_id, std::size_t n_parts, Logger& logger) -> bool {
        std::size_t number_of_tensors{};
        std::size_t total_size {};
        std::string name(64, ' ');

        printf("Loading Model ");

        while(!reader.eof()) {
            std::int32_t n_dims;
            std::int32_t length;
            std::int32_t ftype;

            reader.read(&n_dims);
            reader.read(&length);
            reader.read(&ftype);

            if (reader.eof()) break;

            std::int32_t total_number_of_elements{1};
            std::int32_t extents[2] = { 1, 1 };
            assert(n_dims <= 2 && "number of dimensions should be less than 3");
            for(std::int32_t i {}; i < n_dims; ++i) {
                reader.read(extents + i);
                total_number_of_elements *= extents[i];
            }
            
            name.resize(length);
            reader.read(name.data(), length);
            if (model.tensors.count(name) == 0) {
                logger.log_err("Model", "unkown tensor '", name, "' in model file");
                return false;
            }

            // split_type = 0: split by columns
            // split_type = 1: split by rows
            auto split_type = 0;

            // split_type = 0:
            // regex:
            //   - tok_embeddings.*
            //   - layers.*.attention.wo.weight
            //   - layers.*.feed_forward.w2.weight

            // split_type = 1:
            // regex:
            //   - output.*
            //   - layers.*.attention.wq.weight
            //   - layers.*.attention.wk.weight
            //   - layers.*.attention.wv.weight
            //   - layers.*.feed_forward.w1.weight
            //   - layers.*.feed_forward.w3.weight

            if (name.find("tok_embeddings") != std::string::npos) {
                split_type = 0;
            } else if (name.find("layers") != std::string::npos) {
                if (name.find("attention.wo.weight") != std::string::npos) {
                    split_type = 0;
                } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                    split_type = 0;
                } else {
                    split_type = 1;
                }
            } else if (name.find("output") != std::string::npos) {
                split_type = 1;
            }

            auto tensor = model.tensors[name];

            if (n_dims == 1) {
                if (ggml_nelements(tensor) != total_number_of_elements) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file\n");
                    return false;
                }

                if (tensor->ne[0] != extents[0] || tensor->ne[1] != extents[1]) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file: got [", tensor->ne[0], ", ", tensor->ne[1], "], but expected [", extents[0], ", ", extents[1],"]\n");
                    return false;
                }
            } else {
                if ((ggml_nelements(tensor) / n_parts) != total_number_of_elements) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file\n");
                    return false;
                }

                std::int32_t temp_ne[] = { tensor->ne[0] / (split_type == 0 ? static_cast<int>(n_parts) : 1 ), tensor->ne[1] / (split_type == 0 ? 1 : static_cast<int>(n_parts) ) };
                
                if (temp_ne[0] != extents[0] || temp_ne[1] != extents[1]) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file: got [", temp_ne[0], ", ", temp_ne[1], "], but expected [", extents[0], ", ", extents[1],"]\n");
                    return false;
                }
            }

            std::size_t bpe{};

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(extents[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(extents[0] % 64 == 0); break;
                default: {
                    logger.log_err("Modal", "unknown ftype ", ftype, " in model file\n");
                    return false;
                }
            }

            if (n_dims == 1 || n_parts == 1) {
                if ((total_number_of_elements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file: got ", ggml_nbytes(tensor), ", but expected ", total_number_of_elements * bpe, '\n');
                    return false;
                }

                if (part_id == 0) {
                    reader.read(tensor->data, sizeof(char), ggml_nbytes(tensor));
                } else {
                    reader.seek(ggml_nbytes(tensor), fastllama::BinaryFileReader::SeekReference::Current);
                }

                total_size += ggml_nbytes(tensor);
            } else {
                if ((total_number_of_elements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)/n_parts) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file: got ", ggml_nbytes(tensor), ", but expected ", total_number_of_elements * bpe, '\n');
                    return false;
                }

                if (split_type == 0) {
                    auto const np0 = extents[0];

                    auto const row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                    assert(row_size == tensor->nb[1]);

                    for (int i1 = 0; i1 < np0; ++i1) {
                        auto const offset_row = i1*row_size;
                        auto const offset = offset_row + ((part_id * np0) / ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                        reader.read(static_cast<char*>(tensor->data) + offset, row_size/n_parts);
                    }
                } else {
                    auto const np1 = extents[1];

                    auto const row_size = (tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);

                    for (int i1 = 0; i1 < np1; ++i1) {
                        auto const offset_row = (i1 + part_id * np1)*row_size;
                        reader.read(static_cast<char*>(tensor->data) + offset_row, row_size);
                    }
                }

                total_size += ggml_nbytes(tensor)/n_parts;
            }

            if (++number_of_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }
        printf(" done\n");

        char buff[20] = {0};
        auto len = snprintf(buff, sizeof(buff), "%8.2f", total_size/(1024.0*1024.0));
        logger.log("Model", "model size = ", std::string(buff, len), " MB / num tensors = ", number_of_tensors, "\n");

        return true;
    }

    FASTLLAMA_ALWAYS_INLINE static auto parse_tensor_data(std::string_view filepath, Model& model, std::size_t offset, Logger& logger) -> bool {
        std::size_t n_parts{ model.model_id.number_of_parts };
        auto total_size = filepath.size();
        std::string file_part_path(filepath);
        for(auto i = 0ul; i < n_parts; ++i) {
            auto const part_id = i;
            file_part_path.resize(total_size);
            if (i > 0) {
                file_part_path.push_back('.');
                file_part_path.append(std::to_string(part_id));
            }

            logger.log("Model", "loading model part ", part_id + 1, '/', n_parts, " from ", file_part_path, '\n');

            auto reader = fastllama::BinaryFileReader(file_part_path);
            if (!reader) {
                logger.log_err("Model", "failed to open ", file_part_path, '\n');
                return false;
            }

            if (!reader.seek(offset)) {
                logger.log_err("Model", "failed to seek the data in ", file_part_path, " at the offset ", offset, '\n');
                return false;
            }

            load_model_weights(reader, model, part_id, n_parts, logger);
        }

        return true;
    }

    FASTLLAMA_ALWAYS_INLINE static auto read_header(std::string_view filepath, Model& model, BinaryFileReader& reader, Logger& logger) -> std::optional<std::size_t> {
        if (!verify_magic_number(reader)) {
            logger.log_err("Model", "invalid model file ", filepath, " (bad magic)\n");
            return std::nullopt;
        }

        int n_ff{0};

        // Initialize hyperparameters
        {
            load_hyperparams(reader, model.params);
            n_ff = ((2*(4*model.params.n_embd)/3 + model.params.n_mult - 1)/model.params.n_mult)*model.params.n_mult;

            logger.log("Model", "n_vocab = ", model.params.n_vocab, '\n');
            logger.log("Model", "n_ctx   = ", model.params.n_ctx, '\n');
            logger.log("Model", "n_embd  = ", model.params.n_embd, '\n');
            logger.log("Model", "n_mult  = ", model.params.n_mult, '\n');
            logger.log("Model", "n_head  = ", model.params.n_head, '\n');
            logger.log("Model", "n_layer = ", model.params.n_layer, '\n');
            logger.log("Model", "n_rot   = ", model.params.n_rot, '\n');
            logger.log("Model", "f16     = ", model.params.f16, '\n');
            logger.log("Model", "n_ff    = ", n_ff, '\n');
            logger.log("Model", "n_parts = ", model.model_id.number_of_parts, '\n');
        }

        load_vocab(reader, model.vocabulary, static_cast<std::size_t>(model.params.n_vocab));

        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        ggml_type wtype = GGML_TYPE_COUNT;
        switch (model.params.f16) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            case 2: wtype = GGML_TYPE_Q4_0; break;
            case 3: wtype = GGML_TYPE_Q4_1; break;
            default: {
                logger.log_err("Model", "invalid model file ", filepath, " (bad f16 value ", model.params.f16, ")\n");
                return std::nullopt;
            }
        }

        const ggml_type wtype2 = GGML_TYPE_F32;

        auto & ctx = model.ctx;

        size_t ctx_size = 0;
        
        // Get total context size
        {
            auto const& params = model.params;

            const int n_embd  = params.n_embd;
            const int n_layer = params.n_layer;
            const int n_ctx   = params.n_ctx;
            const int n_vocab = params.n_vocab;

            ctx_size += n_embd*n_vocab * ggml_type_sizef(wtype); // tok_embeddings

            ctx_size += n_embd * ggml_type_sizef(wtype2); // norm

            ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // output

            ctx_size += n_layer * (n_embd * ggml_type_sizef(wtype2)); // attention_norm

            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wq
            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wk
            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wv
            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wo

            ctx_size += n_layer * (n_embd * ggml_type_sizef ( wtype2)); // ffn_norm

            ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w1
            ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w2
            ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w3

            ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(wtype2); // memory_k
            ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(wtype2); // memory_v

            ctx_size += (5 + 10 * n_layer) * 256; // object overhead

            char buff[20] = {0};
            snprintf(buff, 20, "%6.2f", ctx_size/(1024.0*1024.0));
            logger.log("Model", "ggml ctx size = ", std::string(buff), " MB\n");
        }


        // create the ggml context
        {
            ggml_init_params params = {
                .mem_size   = ctx_size,
                .mem_buffer = NULL,
            };

            model.ctx = ggml_init(params);
            if (!model.ctx) {
                logger.log_err("Model", "unable to allocate memory for model\n");
                return std::nullopt;
            }
        }

        prepare_memory_for_weight(model, wtype, wtype2, static_cast<int>(n_ff));

        auto const memory_size = prepare_memory_for_key_value_memory(model);

        {
            char buff[20] = {0};
            auto size = snprintf(buff, sizeof(buff), "%8.2f", memory_size/(1024.0*1024.0));
            logger.log("Model", "memory_size = ", std::string(buff, size), " MB, n_mem = ", model.params.n_layer * model.params.n_ctx, '\n');
        }

        return { reader.tell() };
    }

    auto Model::unload() {
        is_valid = false;
        if(ctx != nullptr) ggml_free(ctx);
        ctx = nullptr;
    }

    bool Model::load(std::string_view model_name, std::string_view filepath) {
        logger.log("Model", "loading model from ", filepath, " - please wait ...\n");
        
        this->is_valid = false;

        // Get model id
        {
            auto temp_model_id = ModelId::from_str_case_insenstive(model_name);
            if (!temp_model_id) {
                logger.log_err("Model", "invalid model id'", model_name, "'\n");
                return false;
            }

            this->model_id = temp_model_id;
        }

        auto reader = fastllama::BinaryFileReader(filepath);
        if (!reader) {
            logger.log_err("Model", "failed to open ", filepath, '\n');
            return false;
        }

        auto maybe_offset = read_header(filepath, *this, reader, logger);
        
        if (!maybe_offset.has_value()) return false;
        auto offset = maybe_offset.value();

        reader.close();

        if (!parse_tensor_data(filepath, *this, offset, logger)) return false;

        this->is_valid = true;
        return true;
    }


    auto Model::eval(
            int threads,
            std::size_t n_past,
            std::vector<vocab_id> const& embd_inp,
            std::vector<float>& embd_w,
            std::size_t& mem_per_token
        ) -> bool
    {
        if (!is_valid) {
            logger.log_err(__func__, "model is not valid\n");
            return false;
        }
        auto const normalized_threads = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), threads));
        auto const embd_size = embd_inp.size();

        auto const n_embd  = params.n_embd;
        auto const n_layer = params.n_layer;
        auto const n_ctx   = params.n_ctx;
        auto const n_head  = params.n_head;
        auto const n_vocab = params.n_vocab;
        auto const n_rot   = params.n_embd / params.n_head;

        auto const d_key = n_embd / n_head;
        
        try {
            m_buffer.resize(m_buffer_size);
        } catch([[maybe_unused]] std::exception& e) {
            logger.log_err(__func__, "failed to allocate ", m_buffer_size, " bytes\n");
            return false;
        }

        auto total_mem_for_token = mem_per_token * embd_size;
        if (mem_per_token > 0 && (total_mem_for_token > m_buffer_size)) {
            // add 10% to account for ggml object overhead
            m_buffer_size = 1.1 * total_mem_for_token;
            try {
                m_buffer.resize(m_buffer_size);
            } catch([[maybe_unused]] std::exception& e) {
                logger.log_err(__func__, "failed to allocate ", m_buffer_size, " bytes\n");
                return false;
            }
        }

        ggml_init_params mem_params;
        mem_params.mem_size   = m_buffer_size;
        mem_params.mem_buffer = reinterpret_cast<void*>(m_buffer.data());

        ggml_context * ctx0 = ggml_init(mem_params);
        ggml_cgraph gf;
        gf.n_threads = normalized_threads;

        ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, embd_size);
        memcpy(embd->data, embd_inp.data(), embd_size * ggml_element_size(embd));

        ggml_tensor* inpL = ggml_get_rows(ctx0, tok_embeddings, embd);

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor* inpSA = inpL;

            ggml_tensor* cur;

            // norm
            {
                cur = ggml_norm(ctx0, inpL);

                // cur = attention_norm*cur
                cur = ggml_mul(ctx0,
                            ggml_repeat(ctx0, layers[il].attention_norm, cur),
                            cur);
            }

            // self-attention
            {
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layers[il].wq, cur);
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layers[il].wk, cur);
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layers[il].wv, cur);

                // store key and value to memory
                if (embd_size >= 1) {
                    ggml_tensor * k = ggml_view_1d(ctx0, memory_k, embd_size*n_embd, (ggml_element_size(memory_k)*n_embd)*(il*n_ctx + n_past));
                    ggml_tensor * v = ggml_view_1d(ctx0, memory_v, embd_size*n_embd, (ggml_element_size(memory_v)*n_embd)*(il*n_ctx + n_past));

                    ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, embd_size).permute(0, 2, 1, 3)
                ggml_tensor * Q =
                    ggml_permute(ctx0,
                            ggml_rope(ctx0,
                                ggml_cpy(ctx0,
                                    Qcur,
                                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, embd_size)),
                                n_past, n_rot, 0),
                            0, 2, 1, 3);

                // K = Kmem.view(n_embd/n_head, n_head, n_past + embd_size).permute(0, 2, 1, 3)
                ggml_tensor * K =
                    ggml_permute(ctx0,
                            ggml_rope(ctx0,
                                ggml_reshape_3d(ctx0,
                                    ggml_view_1d(ctx0, memory_k, (n_past + embd_size)*n_embd, il*n_ctx*ggml_element_size(memory_k)*n_embd),
                                    n_embd/n_head, n_head, n_past + embd_size),
                                n_past, n_rot, 1),
                            0, 2, 1, 3);

                // K * Q
                ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                ggml_tensor * KQ_scaled =
                    ggml_scale(ctx0,
                            KQ,
                            ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                            );

                // KQ_masked = mask_past(KQ_scaled)
                ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + embd_size).permute(1, 2, 0, 3).contiguous()
                ggml_tensor * V_trans =
                    ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, memory_v, (n_past + embd_size)*n_embd, il*n_ctx*ggml_element_size(memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + embd_size),
                            1, 2, 0, 3);

                // KQV = transpose(V) * KQ_soft_max
                ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, embd_size)
                cur = ggml_cpy(ctx0,
                        KQV_merged,
                        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, embd_size));

                // projection (no bias)
                cur = ggml_mul_mat(ctx0,
                        layers[il].wo,
                        cur);
            }

            ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

            // feed-forward network
            {
                // norm
                {
                    cur = ggml_norm(ctx0, inpFF);

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

            cur  = ggml_add(ctx0, cur, inpFF);

            // input for next layer
            inpL = cur;
        }

        // norm
        {
            inpL = ggml_norm(ctx0, inpL);

            // inpL = norm*inpL
            inpL = ggml_mul(ctx0,
                        ggml_repeat(ctx0, norm, inpL),
                        inpL);
        }

        // lm_head
        {
            inpL = ggml_mul_mat(ctx0, output, inpL);
        }

        // logits -> probs
        //inpL = ggml_soft_max(ctx0, inpL);

        // run the computation
        ggml_build_forward_expand(&gf, inpL);
        ggml_graph_compute       (ctx0, &gf);

        // return result for just the last token
        embd_w.resize(static_cast<std::size_t>(n_vocab));
        memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(embd_size-1)), sizeof(float)*n_vocab);

        if (mem_per_token == 0) {
            mem_per_token = ggml_used_mem(ctx0) / embd_size;
        }

        ggml_free(ctx0);

        return true;
    }

} // namespace fastllama
