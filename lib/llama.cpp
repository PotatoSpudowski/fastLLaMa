#include "llama.hpp"
#include "file_reader.hpp"

namespace fastllama {
    
    constexpr auto verify_magic_number(BinaryFileReader& reader) noexcept -> bool {
        std::uint32_t magic{};
        reader.read(&magic);
        return magic == magic_number_v;
    }

    constexpr auto load_hyperparams(BinaryFileReader& reader, HyperParams& params) {
        reader.read(&params.n_vocab);
        reader.read(&params.n_embd);
        reader.read(&params.n_mult);
        reader.read(&params.n_head);
        reader.read(&params.n_layer);
        reader.read(&params.n_rot);
        reader.read(&params.f16);
    }

    auto load_vocab(BinaryFileReader& reader, Vocab& vocab, std::size_t size) {
        vocab.id_to_token.resize(size);
        std::string word(' ', 64);

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

    auto prepare_memory_for_weight(Model& model, ggml_type wtype, ggml_type wtype2, int n_ff) {
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
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wq.weight.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wq;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wk.weight.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wq;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wv.weight.weight", i);
            model.tensors[std::string(temp_buff, str_size)] = layer.wv;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wo.weight.weight", i);
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

    auto prepare_memory_for_key_value_memory(Model& model) -> std::size_t {
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

    auto parse_tensor_data(std::string_view filepath, Model& model, Logger& logger) {
        std::size_t n_part{ model.model_id.number_of_parts };
        std::string file_part_path('\0', filepath.size() + 20);
        for(auto i = 0ul; i < n_part + 1; ++i) {
            auto const part_id = i;
            auto size = snprintf(file_part_path.data(), file_part_path.size(), "%.*s.%zu", static_cast<int>(filepath.size()), filepath.data(), part_id);
            file_part_path.resize(static_cast<std::size_t>(size));
            logger.log("Model", file_part_path, '\n');
        }
    }

    auto read_header(std::string_view filepath, Model& model, BinaryFileReader& reader, Logger& logger) -> std::optional<std::size_t> {
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
            struct ggml_init_params params = {
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

    Model::Model(std::string_view model_name, std::string_view filepath, std::size_t context_size, Logger logger)
    {
        logger.log("Model", "loading model from ", filepath, " - please wait ...\n");
        
        this->is_valid = false;
        this->params.n_ctx = context_size;

        // Get model id
        {
            auto temp_model_id = ModelId::from_str_case_insenstive(model_name);
            if (!temp_model_id) {
                logger.log_err("Model", "invalid model id'", model_name, "'\n");
                return;
            }

            this->model_id = temp_model_id;
        }

        auto reader = fastllama::BinaryFileReader(filepath);
        if (!reader) {
            logger.log_err("Model", "failed to open ", filepath, '\n');
            return;
        }

        auto maybe_offset = read_header(filepath, *this, reader, logger);
        
        if (!maybe_offset.has_value()) return;
        auto offset = maybe_offset.value();

        reader.close();

        parse_tensor_data(filepath, *this, logger);

        this->is_valid = true;
    }

} // namespace fastllama
