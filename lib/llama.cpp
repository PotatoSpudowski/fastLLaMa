#include "llama.hpp"
#include "file_reader.hpp"
#include "file_pipe.hpp"
#include <cassert>
#include "macro.hpp"
#include <fstream>
#include <numeric>
#include <functional>
#include "span.hpp"

namespace fastllama {

    FASTLLAMA_ALWAYS_INLINE static constexpr auto verify_magic_number(BinaryFileReader& reader) noexcept -> bool {
        std::uint32_t magic{};
        reader.read(&magic);
        return magic == magic_number_v;
    }
    
    FASTLLAMA_ALWAYS_INLINE static constexpr auto verify_file_version(BinaryFileReader& reader, std::uint32_t* format_version = nullptr) noexcept -> bool {
        std::uint32_t version{};
        reader.read(&version);
        if (format_version != nullptr) *format_version = version;
        return version == file_version_v;
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

    FASTLLAMA_ALWAYS_INLINE static auto load_vocab(BinaryFileReader& reader, Vocab& vocab, std::size_t size, bool has_padding, bool is_old_model) {
        vocab.id_to_token.resize(size);
        std::string word(64, ' ');

        auto new_size = size - static_cast<std::size_t>(has_padding);
        for (auto i = 0ul; i < new_size; ++i) {
            std::uint32_t len;
            reader.read(&len);

            word.resize(len);
            reader.read(word.data(), len);

            float score{};
            if (!is_old_model) reader.read(&score);
            vocab.set_word(static_cast<typename Vocab::id>(i), word, score);
        }

        if (!has_padding) return;

        std::string pad_token = "<pad>";
        vocab.set_word(static_cast<typename Vocab::id>(new_size), std::move(pad_token), 0);
    }

    FASTLLAMA_ALWAYS_INLINE static auto prepare_memory_for_weight(Model& model, ggml_type vtype, ggml_type wtype, int n_ff) {
        auto const& params  = model.params;
        auto const  n_embd  = params.n_embd;
        auto const  n_layer = params.n_layer;
        // auto const  n_ctx   = params.n_ctx;
        auto const  n_vocab = params.n_vocab;
        auto&       ctx     = model.ctx;

        model.layers.resize(static_cast<std::size_t>(n_layer));
        model.tok_embeddings    = ggml_new_tensor_2d(ctx, vtype, n_embd, n_vocab);
        model.norm              = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output            = ggml_new_tensor_2d(ctx, vtype, n_embd, n_vocab);

        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;

        model.tensors["norm.weight"]   = model.norm;
        model.tensors["output.weight"] = model.output;

        char temp_buff[128] = {0};

        for(auto i = 0ul; i < static_cast<std::size_t>(n_layer); ++i) {
            auto& layer = model.layers[i];
            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);

            auto str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention_norm.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.attention_norm;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wq.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.wq;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wk.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.wk;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wv.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.wv;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.attention.wo.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.wo;
            
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.ffn_norm.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.ffn_norm;
            
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.feed_forward.w1.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.w1;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.feed_forward.w2.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.w2;
            
            str_size = snprintf(temp_buff, sizeof(temp_buff), "layers.%zu.feed_forward.w3.weight", i);
            model.tensors[std::string(temp_buff, static_cast<std::size_t>(str_size))] = layer.w3;
        }
    }

    FASTLLAMA_ALWAYS_INLINE static auto load_model_weights(BinaryFileReader& reader, Model& model, std::size_t part_id, std::size_t n_parts, Logger& logger, bool is_old_model) -> bool {
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

            std::size_t total_number_of_elements{1};
            std::int32_t extents[2] = { 1, 1 };
            
            assert(n_dims <= 2 && "number of dimensions should be less than 3");

            for(std::int32_t i {}; i < n_dims; ++i) {
                reader.read(extents + i);
                total_number_of_elements *= static_cast<std::size_t>(extents[i]);
            }
            
            auto str_len = static_cast<std::size_t>(length);
            name.resize(str_len);
            reader.read(name.data(), str_len);
            if (model.tensors.count(name) == 0) {
                logger.log_err("Model", "unkown tensor '", name, "' in model file\n");
                return false;
            }

            if (!is_old_model) {
                // ensure tensor data is aligned
                auto maybe_offset = reader.tell();
                if (!maybe_offset) return false;
                auto offset = *maybe_offset;
                offset = (offset + 31ul) & static_cast<std::size_t>(-32);
                reader.seek(offset);
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
                if (static_cast<std::size_t>(ggml_nelements(tensor)) != total_number_of_elements) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file\n");
                    return false;
                }

                if (tensor->ne[0] != extents[0] || tensor->ne[1] != extents[1]) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file: got [", tensor->ne[0], ", ", tensor->ne[1], "], but expected [", extents[0], ", ", extents[1],"]\n");
                    return false;
                }
            } else {
                if ((static_cast<std::size_t>(ggml_nelements(tensor)) / n_parts) != total_number_of_elements) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file\n");
                    return false;
                }

                std::int64_t temp_ne[] = { tensor->ne[0] / (split_type == 0 ? static_cast<int>(n_parts) : 1 ), tensor->ne[1] / (split_type == 0 ? 1 : static_cast<int>(n_parts) ) };
                
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
                if ((total_number_of_elements * bpe) / static_cast<std::size_t>(ggml_blck_size(tensor->type)) != ggml_nbytes(tensor)) {
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
                if ((total_number_of_elements * bpe) / static_cast<std::size_t>(ggml_blck_size(tensor->type)) != ggml_nbytes(tensor)/n_parts) {
                    logger.log_err("Model", "tensor '", name, "' has wrong size in model file: got ", ggml_nbytes(tensor), ", but expected ", total_number_of_elements * bpe, '\n');
                    return false;
                }

                if (split_type == 0) {
                    auto const np0 = static_cast<std::size_t>(extents[0]);

                    auto const row_size = static_cast<std::size_t>(tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                    assert(row_size == tensor->nb[1]);

                    for (auto i1 = 0ul; i1 < np0; ++i1) {
                        auto const offset_row = i1*row_size;
                        auto const offset = offset_row + ((part_id * np0) / static_cast<std::size_t>(ggml_blck_size(tensor->type)))*ggml_type_size(tensor->type);
                        reader.read(static_cast<char*>(tensor->data) + offset, row_size/n_parts);
                    }
                } else {
                    auto const np1 = static_cast<std::size_t>(extents[1]);

                    auto const row_size = static_cast<std::size_t>(tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);

                    for (auto i1 = 0ul; i1 < np1; ++i1) {
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
        logger.log("Model", "model size = ", std::string(buff, static_cast<std::size_t>(len)), " MB / num tensors = ", number_of_tensors, "\n");

        return true;
    }

    FASTLLAMA_ALWAYS_INLINE static auto parse_tensor_data(std::vector<char>& file_buffer, std::string_view filepath, Model& model, std::size_t offset, Logger& logger, bool is_old_model) -> bool {
        std::size_t n_parts{ model.model_id.config.number_of_parts };
        
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

            reader.set_buffer(file_buffer.data(), file_buffer.size());

            if (!reader.seek(offset)) {
                logger.log_err("Model", "failed to seek the data in ", file_part_path, " at the offset ", offset, '\n');
                return false;
            }

            load_model_weights(reader, model, part_id, n_parts, logger, is_old_model);
        }

        return true;
    }

    FASTLLAMA_ALWAYS_INLINE static auto read_header(
        std::string_view filepath,
        Model& model,
        BinaryFileReader& reader,
        Logger& logger,
        bool is_old_model
    ) -> std::optional<std::size_t> {
        using namespace ::fastllama::literals;

        if (!verify_magic_number(reader)) {
            logger.log_err("Model", "invalid model file ", filepath, " (bad magic)\n");
            return std::nullopt;
        }

        std::uint32_t format_version{};
        if (!is_old_model && !verify_file_version(reader, &format_version)) {
            logger.log_err("Model", "invalid  model file ", filepath, "(unsupported format version ", format_version, " expected ", file_version_v, "\n");
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
            logger.log("Model", "n_parts = ", model.model_id.config.number_of_parts, '\n');
        }

        load_vocab(reader, model.vocabulary, static_cast<std::size_t>(model.params.n_vocab), model.model_id.config.has_vocab_padding, is_old_model);
        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        // wtype is for per-layer weights, while vtype is for other weights

        ggml_type wtype, vtype;
        switch (model.params.f16) {
            case 0: wtype = vtype = GGML_TYPE_F32;  break;
            case 1: wtype = vtype = GGML_TYPE_F16;  break;
            case 2: wtype = vtype = GGML_TYPE_Q4_0; break;
            case 3: wtype = vtype = GGML_TYPE_Q4_1; break;
            case 4: wtype = GGML_TYPE_Q4_1; vtype = GGML_TYPE_F16; break;
            default: {
                logger.log_err("Model", "invalid model file ", filepath, " (bad f16 value ", model.params.f16, ")\n");
                return std::nullopt;
            }
        }

        float ctx_size = 0;
        
        // Get total context size
        {
            auto const& params = model.params;

            const int n_embd  = params.n_embd;
            const int n_layer = params.n_layer;
            const int n_ctx   = params.n_ctx;
            const int n_vocab = params.n_vocab;

            ctx_size += n_embd*n_vocab * ggml_type_sizef(vtype); // tok_embeddings

            ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // norm

            ctx_size += n_embd * n_vocab * ggml_type_sizef(vtype); // output

            ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // attention_norm

            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wq
            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wk
            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wv
            ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // wo

            ctx_size += n_layer * (n_embd * ggml_type_sizef ( GGML_TYPE_F32)); // ffn_norm

            ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w1
            ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w2
            ctx_size += n_layer * (n_ff * n_embd * ggml_type_sizef(wtype)); // w3

            ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // kv_self.k
            ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // kv_self.v

            ctx_size += (5 + 10 * n_layer) * 256; // object overhead

            char buff[20] = {0};
            snprintf(buff, 20, "%6.2Lf", static_cast<long double>(ctx_size)/1.0_MiB);
            logger.log("Model", "ggml ctx size = ", std::string(buff), " MB\n");
        }


        // create the ggml context
        {
            model.buffer.resize(static_cast<std::size_t>(std::ceil(ctx_size)));
            ggml_init_params params {};
            params.mem_size = model.buffer.size();
            params.mem_buffer = model.buffer.data();

            model.ctx = ggml_init(params);
            if (!model.ctx) {
                logger.log_err("Model", "unable to allocate memory for model\n");
                return std::nullopt;
            }
        }

        prepare_memory_for_weight(model, vtype, wtype, static_cast<int>(n_ff));

        {
            std::size_t const scale = model.kv_self.memory_type == GGML_TYPE_F32 ? 2 : 1;


            // this is the total memory required to run the inference
            auto const mem_required =
                ctx_size +
                model.model_id.config.mem_required_for_scratch_buff_0 +
                model.model_id.config.mem_required_for_scratch_buff_1 +
                model.model_id.config.mem_required_for_eval;

            // this is the memory required by one llama_state
            auto const mem_required_state = model.model_id.config.mem_required_for_kv_self_buff * scale;

            char buff[20] = {0};
            std::string mem_req_str, mem_req_state_str;
            {
                auto const len = snprintf(buff, sizeof(buff), "%7.2Lf", static_cast<long double>(mem_required) / 1.0_MiB);
                mem_req_str = std::string(buff, static_cast<std::size_t>(len));
            }
            {
                auto const len = snprintf(buff, sizeof(buff), "%7.2Lf", static_cast<long double>(mem_required_state) / 1.0_MiB);
                mem_req_state_str = std::string(buff, static_cast<std::size_t>(len));
            }

            logger.log("Model", "mem required  = ", mem_req_str, " MB (+ ", mem_req_state_str, " MB per state)\n");
        }

        return { reader.tell() };
    }

    auto Model::unload() -> void {
        is_valid = false;
        kv_self.deinit();
        if(ctx) ggml_free(ctx);
        ctx = nullptr;
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

        ggml_init_params mem_params;
        mem_params.mem_size   = buffer.size();
        mem_params.mem_buffer = buffer.data();
        mem_params.no_alloc   = false;

        this->ctx = ggml_init(mem_params);

        if (!this->ctx) {
            logger.log_err("KVCacheBuffer::init", "failed to allocate memory for kv cache\n");
            return false;
        }

        this->k = ggml_new_tensor_1d(this->ctx, memory_type, static_cast<std::int64_t>(number_of_elements));
        this->v = ggml_new_tensor_1d(this->ctx, memory_type, static_cast<std::int64_t>(number_of_elements));

        auto const total_kv_size = ggml_nbytes(this->k) + ggml_nbytes(this->v);

        char buff[20] = {0};
        auto const len = snprintf(buff, sizeof(buff), "%7.2f", total_kv_size / (1024.0 * 1024.0));

        logger.log("KVCacheBuffer::init", "kv self size  = ", std::string_view{buff, static_cast<std::size_t>(len)}, " MB\n");
        
        return true;
    }

    void KVCacheBuffer::deinit([[maybe_unused]] Logger const& logger) {
        if (!ctx) return;
        ggml_free(ctx);
        ctx = nullptr;
    }

    bool Model::dump_vocab(std::string_view filepath) {
        std::ofstream f(filepath);
        if (!f) return false;

        for(auto i = 0ul; i < vocabulary.id_to_token.size(); ++i) {
            f << i << " : " << vocabulary.id_to_token[i].tok<<'\n';
        }
        f.close();
        return true;
    }

    bool Model::load(std::string_view model_name, std::string_view filepath, bool is_old_model) {
        using namespace ::fastllama::literals;

        logger.log("Model", "loading model(='", model_name,"') from ", filepath, " - please wait ...\n");
        
        this->is_valid = false;

        // Initialize cache
        if (!kv_self.init(params, logger)) return false;

        // Get model id
        {
            auto temp_model_id = ModelId::from_str_case_insensitive(model_name);
            if (!temp_model_id) {
                logger.log_err("Model", "invalid model id'", model_name, "'\n");
                return false;
            }

            this->model_id = temp_model_id;
        }

        // Initialize compute buffers
        {
            model_id.config.mem_required_for_eval += static_cast<std::size_t>(n_batch) * 20_MiB; // extra space large batch
            buf_compute.resize(model_id.config.mem_required_for_eval);
            buf_scratch[0].resize(model_id.config.mem_required_for_scratch_buff_0);
            buf_scratch[1].resize(model_id.config.mem_required_for_scratch_buff_1);
        }

        auto reader = fastllama::BinaryFileReader(filepath);
        if (!reader) {
            logger.log_err("Model", "failed to open ", filepath, '\n');
            return false;
        }

        std::vector<char> file_buffer((1 << 20));
        reader.set_buffer(file_buffer.data(), file_buffer.size());

        auto maybe_offset = read_header(filepath, *this, reader, logger, is_old_model);
        
        if (!maybe_offset.has_value()) return false;
        auto offset = maybe_offset.value();

        reader.close();

        if (!parse_tensor_data(file_buffer, filepath, *this, offset, logger, is_old_model)) return false;

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
        auto const N = embd_inp.size();

        auto const n_embd  = params.n_embd;
        auto const n_layer = params.n_layer;
        auto const n_ctx   = params.n_ctx;
        auto const n_head  = params.n_head;
        auto const n_vocab = params.n_vocab;
        auto const n_rot   = params.n_embd / params.n_head;

        ggml_init_params mem_params {};
        mem_params.mem_size   = buf_compute.size();
        mem_params.mem_buffer = reinterpret_cast<void*>(buf_compute.data());

        ggml_context * ctx0 = ggml_init(mem_params);
        ggml_cgraph gf{};
        gf.n_threads = (N >= 32 && ggml_cpu_has_blas() ? 1 : threads);

        ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, static_cast<std::int64_t>(N));
        memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

        ggml_tensor* inpL = ggml_get_rows(ctx0, tok_embeddings, embd);
        
        for (auto il = 0ul; il < static_cast<std::size_t>(n_layer); ++il) {
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
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, layers[il].wq, cur);
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, layers[il].wk, cur);
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, layers[il].wv, cur);

                // store key and value to memory
                if (N >= 1) {
                    auto const ne0 = static_cast<std::int64_t>(N * static_cast<std::size_t>(n_embd));
                    ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, ne0, (ggml_element_size(kv_self.k)*static_cast<std::size_t>(n_embd))*(il*static_cast<std::size_t>(n_ctx) + n_past));
                    ggml_tensor * v = ggml_view_1d(ctx0, kv_self.v, ne0, (ggml_element_size(kv_self.v)*static_cast<std::size_t>(n_embd))*(il*static_cast<std::size_t>(n_ctx) + n_past));

                    ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                ggml_tensor * Q =
                    ggml_permute(ctx0,
                            ggml_rope(ctx0,
                                ggml_cpy(ctx0,
                                    Qcur,
                                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, static_cast<std::int64_t>(N))),
                                static_cast<int>(n_past), n_rot, 0),
                            0, 2, 1, 3);

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                ggml_tensor * K =
                    ggml_permute(ctx0,
                            ggml_rope(ctx0,
                                ggml_reshape_3d(ctx0,
                                    ggml_view_1d(ctx0, kv_self.k, static_cast<std::int64_t>((n_past + N)*static_cast<std::size_t>(n_embd)), il*static_cast<std::size_t>(n_ctx)*ggml_element_size(kv_self.k)*static_cast<std::size_t>(n_embd)),
                                    n_embd/n_head, n_head, static_cast<std::int64_t>(n_past + N)),
                                static_cast<int>(n_past), n_rot, 1),
                            0, 2, 1, 3);

                // K * Q
                ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                ggml_tensor * KQ_scaled =
                    ggml_scale(ctx0,
                            KQ,
                            ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));

                // KQ_masked = mask_past(KQ_scaled)
                ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, static_cast<int>(n_past));

                // KQ = soft_max(KQ_masked)
                ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
                ggml_tensor * V_trans =
                    ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                                ggml_reshape_3d(ctx0,
                                    ggml_view_1d(ctx0, kv_self.v, static_cast<std::int64_t>((n_past + N)*static_cast<std::size_t>(n_embd)), il*static_cast<std::size_t>(n_ctx)*ggml_element_size(kv_self.v)*static_cast<std::size_t>(n_embd)),
                                    n_embd/n_head, n_head, static_cast<std::int64_t>(n_past + N)),
                                1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, kv_self.v->type, static_cast<std::int64_t>(n_past + N), n_embd/n_head, n_head));

                // KQV = transpose(V) * KQ_soft_max
                ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                cur = ggml_cpy(ctx0,
                        KQV_merged,
                        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, static_cast<std::int64_t>(N)));

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

    bool quantize(std::string_view in_filepath, std::string_view out_filepath, int itype) {
        using namespace ::fastllama::literals;

        ggml_type type = GGML_TYPE_Q4_1;

        switch (itype) {
            case 2: type = GGML_TYPE_Q4_0; break;
            case 3: type = GGML_TYPE_Q4_1; break;
            default: fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype); return false;
        };

        if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1) {
            fprintf(stderr, "%s: invalid quantization type %d\n", __func__, type);
            return false;
        }

        printf("%s: loading model from '%.*s'\n", __func__, static_cast<int>(in_filepath.size()), in_filepath.data());

        BinaryFilePipe pipe(in_filepath, out_filepath);
        if (!pipe.get_reader()) {
            fprintf(stderr, "%s: failed to open '%.*s' for reading\n", __func__, static_cast<int>(in_filepath.size()), in_filepath.data());
            return false;
        }
        
        if (!pipe.get_writer()) {
            fprintf(stderr, "%s: failed to open '%.*s' for writing\n", __func__, static_cast<int>(out_filepath.size()), out_filepath.data());
            return false;
        }

        if (!verify_magic_number(pipe.get_reader())) {
            fprintf(stderr, "%s: invalid model file '%.*s' (bad magic)\n", __func__, static_cast<int>(in_filepath.size()), in_filepath.data());
            return false;
        }

        pipe.get_writer().write(&magic_number_v);
        
        std::uint32_t format_version{};
        if (!verify_file_version(pipe.get_reader(), &format_version)) {
            fprintf(stderr, "%s: invalid model file '%.*s' (unsupported format version %zu, expected %d)\n",
                    __func__, static_cast<int>(in_filepath.size()), in_filepath.data(), static_cast<std::size_t>(format_version), file_version_v);
            return false;
        }

        pipe.get_writer().write(&format_version);

        std::string_view parent_fn_name = __func__;

        auto param_print_fn = [parent_fn_name](std::string_view s) {
            return [s, parent_fn_name](auto* val) {
                printf("%.*s: %.*s = %d\n", static_cast<int>(parent_fn_name.size()), parent_fn_name.data(), static_cast<int>(s.size()), s.data(), *val);
                return val;
            };
        };

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
                auto pos = (name.size() > temp_w_str.size() ? name.size() - temp_w_str.size() : 0);
                auto find_pos = name.find(temp_w_str, pos);
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
                        #pragma omp parallel for if (total_number_of_elements > 512)
                        for (auto i = 0ul; i < total_number_of_elements; ++i) {
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

} // namespace fastllama
