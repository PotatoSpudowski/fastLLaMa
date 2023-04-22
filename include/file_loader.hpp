#if !defined(FAST_LLAMA_FILE_LOADER_HPP)
#define FAST_LLAMA_FILE_LOADER_HPP

#include "file_reader.hpp"
#include "file_writer.hpp"
#include "vocab.hpp"
#include "logger.hpp"
#include "tensor/utils.hpp"
#include "llama.hpp"
#include "tensor/mem_context.hpp"
#include "mmap.hpp"
#include "uninitialized_buffer.hpp"

namespace fastllama {

    constexpr std::string_view to_string_view(FileVersion version) noexcept {
        switch(version) {
            case FileVersion::GGML: return "'ggml' (old version with low tokenizer quality and no mmap support)";
            case FileVersion::GGMF_V1: return "ggmf v1 (old version with no mmap support)";
            case FileVersion::GGJT_V1: return "ggjt v1 (latest)";
            default: FAST_LLAMA_ASSERT(false, "Invalid file version");
        }
    }
    
    constexpr std::string_view to_string_view(MagicKind kind) noexcept {
        switch(kind) {
            case MagicKind::GGML: return "ggml";
            case MagicKind::GGMF: return "ggmf";
            case MagicKind::GGLA: return "ggla";
            case MagicKind::GGJT: return "ggjt";
            default: FAST_LLAMA_ASSERT(false, "Invalid magic kind");

        }
    }

    struct FileLoader {
        BinaryFileReader    reader;
        MagicKind           magic_kind{MagicKind::Unknown};
        FileVersion         version{FileVersion::GGML};
        HyperParams         hyperparams;
        LoraAdapterParams   lora_adapter_params;
        Vocab               vocab;
        Logger const*       logger;
        bool                is_failed_to_read{false};

        FileLoader(
            std::string_view filepath,
            std::size_t file_idx,
            TensorsMapping& tensors_map,
            Logger const* logger = nullptr
        ) noexcept
            : reader(filepath)
            , logger(logger ? logger : &Logger::get_null_logger())
        {
            if (!reader) {
                logger->log_err(__func__, "Failed to open file: '", filepath, "'\n");
                is_failed_to_read = true;
                return;
            }

            if (!read_magic_number()) {
                is_failed_to_read = true;
                return;
            }

            if (magic_kind != MagicKind::GGLA) {
                if (!read_hyperparams()) {
                    is_failed_to_read = true;
                    return;
                }

                read_vocab();
            } else {
                read_lora_params();
            }

            read_tensor_metadata(file_idx, tensors_map);

        }

        FileLoader(FileLoader const&) = delete;
        FileLoader(FileLoader&&) noexcept = default;
        FileLoader& operator=(FileLoader const&) = delete;
        FileLoader& operator=(FileLoader&&) noexcept = default;
        ~FileLoader() = default;

    private:

        auto read_magic_number() noexcept -> bool {
            std::uint32_t magic = reader.read_u32();
            FileVersion file_version = FileVersion::GGML;

            if (static_cast<std::uint32_t>(MagicKind::GGML) != magic) {
                if (!read_file_version(&file_version)) {
                    logger->log_err(__func__,
                        "invalid  model file ",
                        reader.path(),
                        "(unsupported format version '", static_cast<int>(version), "', but expected it to between 0 and ",
                        static_cast<int>(FileVersion::Size)
                        ,")\n"
                    );
                    return false;
                }
            }


            switch(magic) {
                case static_cast<std::uint32_t>(MagicKind::GGML):
                    magic_kind = MagicKind::GGML;
                    break;
                case static_cast<std::uint32_t>(MagicKind::GGMF):
                    magic_kind = MagicKind::GGMF;
                    break;
                case static_cast<std::uint32_t>(MagicKind::GGJT):
                    magic_kind = MagicKind::GGJT;
                    break;
                case static_cast<std::uint32_t>(MagicKind::GGLA):
                    magic_kind = MagicKind::GGLA;
                default: {
                    logger->log_err(__func__, "invalid model file ", reader.path(), " (bad magic)\n");
                    return false;
                }
            }

            if (magic_kind == MagicKind::GGLA) return true;

            if (magic_kind == MagicKind::GGML && file_version == FileVersion::GGML) {
                version = FileVersion::GGML;
                return true;
            } else if (magic_kind == MagicKind::GGMF && file_version == FileVersion::GGMF_V1) {
                version = FileVersion::GGMF_V1;
                return true;
            } else if (magic_kind == MagicKind::GGJT && file_version == FileVersion::GGMF_V1) {
                version = FileVersion::GGJT_V1;
                return true;
            }

            char buff[256];
            logger->log_err(__func__, format_str(buff, "unknown (magic, version) combination: %08x, %08x; is this really a GGML file?",
                        magic, static_cast<std::uint32_t>(file_version)));
            return false;
        }

        auto read_file_version(FileVersion* file_version) noexcept -> bool {
            auto version = reader.read_u32();

            switch(version) {
                case static_cast<std::uint32_t>(FileVersion::GGML):
                    *file_version = FileVersion::GGML;
                    return true;
                case static_cast<std::uint32_t>(FileVersion::GGMF_V1):
                    *file_version = FileVersion::GGMF_V1;
                    return true;
                case static_cast<std::uint32_t>(FileVersion::GGJT_V1):
                    *file_version = FileVersion::GGJT_V1;
                    return true;
                default:
                    return false;
            }
        }

        auto read_hyperparams() noexcept -> bool {
            auto res = reader.read(&hyperparams.n_vocab)   &&
                    reader.read(&hyperparams.n_embd)    && 
                    reader.read(&hyperparams.n_mult)    && 
                    reader.read(&hyperparams.n_head)    && 
                    reader.read(&hyperparams.n_layer)   && 
                    reader.read(&hyperparams.n_rot)     && 
                    reader.read(&hyperparams.ftype);
            if (!res) {
                logger->log_err(__func__, "Failed to read hyper parameters from file: ", reader.path(), "\n");
                return false;
            }
            return true;
        }

        auto read_lora_params() noexcept -> bool {
            lora_adapter_params.use_cache_matrix = reader.read_u8();
            if (!lora_adapter_params.use_cache_matrix) {
                lora_adapter_params.r = reader.read_u32();
                lora_adapter_params.alpha = reader.read_f32();
            }
            return true;
        }

        auto read_vocab() -> void {
            auto size = static_cast<std::size_t>(hyperparams.n_vocab);
            vocab.id_to_token.resize(size);

            for (auto i = 0ul; i < size; ++i) {
                if (reader.eof()) {
                    logger->log_err(__func__, "Failed to read vocab from file: ", reader.path(), "\n");
                    return;
                }
                auto len = reader.read_u32();
                auto word = reader.read_string(len);

                float score = (version >= FileVersion::GGMF_V1 ? reader.read_f32() : 0.0f);
                vocab.set_word(static_cast<typename Vocab::id_type>(i), std::move(word), score);
            }
        }

        auto read_tensor_metadata(size_t file_idx, TensorsMapping& tensors_map) -> bool {
            while(!reader.eof() && reader.tell() < reader.size()) {
                TensorShard shard;
                auto n_dims = reader.read_u32();
                auto name_len = reader.read_u32();
                shard.type = static_cast<ggml_type>(reader.read_u32());

                shard.extents.resize(n_dims);

                reader.read(shard.extents.data(), n_dims);

                std::string name = reader.read_string(name_len);

                if (n_dims < 1 || n_dims > 2) {
                    logger->log_err(__func__, "tensor '", name,"' should not be ", n_dims, "-dimensional\n");
                    return false;
                }

                if (!(shard.type >= GGML_TYPE_F32 && shard.type <= GGML_TYPE_Q4_3)) {
                    logger->log_err(__func__, "unrecognized tensor type '", shard.type, "'\n");
                    return false;
                }
                
                if (version >= FileVersion::GGJT_V1) reader.seek(-reader.tell() & 31);
                shard.file_idx = file_idx;
                shard.file_off = reader.tell();

                if (!shard.calc_size(*logger)) return false;
                reader.seek(shard.size, BinaryFileReader::SeekReference::Current);

                std::size_t idx{};

                auto it = tensors_map.tensor_names.find(name);
                if (it == tensors_map.tensor_names.end()) {
                    tensors_map.tensors.emplace_back(name);
                    idx = tensors_map.tensors.size() - 1;
                    tensors_map.tensor_names[name] = idx;
                } else {
                    idx = it->second;
                }

                tensors_map.tensors[idx].shards.push_back(shard);
            }
            return true;
        }
    };

    struct FileSaver {
        BinaryFileWriter    writer;
        FileLoader*         loader;
        Logger const*       logger;
        bool                is_write_failed{false};

        FileSaver(std::string_view path, FileLoader* loader, FType new_ftype, Logger const* logger = nullptr) noexcept
            : writer(path)
            , loader(loader)
            , logger(logger ? logger : &Logger::get_null_logger())
        {
            if (!writer) {
                logger->log_err(__func__, "Failed to open file: '", path, "'\n");
                is_write_failed = true;
                return;
            }

            if (!write_magic()) {
                is_write_failed = true;
                return;
            }

            if (loader->magic_kind != MagicKind::GGLA) {
                if (!write_hyperparams()) {
                    is_write_failed = true;
                    return;
                }

                if (!write_vocab()) {
                    is_write_failed = true;
                    return;
                }
            } else {
                if (!write_lora_adapter()) {
                    is_write_failed = true;
                    return;
                }
            }

        }

        FileSaver(FileSaver const&) = delete;
        FileSaver(FileSaver&&) noexcept = default;
        FileSaver& operator=(FileSaver const&) = delete;
        FileSaver& operator=(FileSaver&&) noexcept = default;
        ~FileSaver() = default;

        auto write_magic() noexcept -> bool {
            auto magic = static_cast<std::uint32_t>(MagicKind::GGJT);
            auto version = 1ul;
            auto const res = writer.write(&magic) && writer.write_u32(version);
            if (!res) {
                logger->log_err(__func__, "Failed to write magic('", magic,"') and version('", version,"') to file: ", writer.path(), "\n");
                return false;
            }
            return true;
        }

        auto write_lora_adapter() noexcept -> bool {
            auto const res = writer.write_u8(loader->lora_adapter_params.use_cache_matrix);
            if (!res) {
                logger->log_err(__func__, "Failed to write lora adapter params to file: '", writer.path(), "'\n");
                return false;
            }
            if (!loader->lora_adapter_params.use_cache_matrix) {
                auto const res = writer.write_u32(loader->lora_adapter_params.r) && writer.write_f32(loader->lora_adapter_params.alpha);
                if (!res) {
                    logger->log_err(__func__, "Failed to write lora adapter params to file: '", writer.path(), "'\n");
                    return false;
                }
            }
            return true;
        }

        auto write_hyperparams() noexcept -> bool {
            auto const res = writer.write(&loader->hyperparams.n_vocab)   &&
                    writer.write(&loader->hyperparams.n_embd)    && 
                    writer.write(&loader->hyperparams.n_mult)    && 
                    writer.write(&loader->hyperparams.n_head)    && 
                    writer.write(&loader->hyperparams.n_layer)   && 
                    writer.write(&loader->hyperparams.n_rot)     && 
                    writer.write(&loader->hyperparams.ftype);
            if (!res) {
                logger->log_err(__func__, "Failed to write hyper parameters to file: '", writer.path(), "'\n");
                return false;
            }
            return true;
        }

        auto write_vocab() -> bool {
            auto size = static_cast<std::size_t>(loader->hyperparams.n_vocab);
            FAST_LLAMA_ASSERT(size == loader->vocab.id_to_token.size(), "vocab size mismatch");
            for (auto i = 0ul; i < size; ++i) {
                auto const& word = loader->vocab.id_to_token[i];
                auto res = writer.write_string(word.tok) && writer.write(&word.score);
                if (!res) {
                    logger->log_err(__func__, "Failed to write vocab to file: '", writer.path(), "'\n");
                    return false;
                }
            }
            return true;
        }

        auto write_tensor(TensorLoader & tensor, ggml_type new_type, const void * new_data, size_t new_size) -> bool {
            FAST_LLAMA_ASSERT(new_type >= GGML_TYPE_F32 && new_type <= GGML_TYPE_Q4_3, "invalid tensor type");

            writer.write_u32(tensor.extents.size());
            writer.write_u32(tensor.name.size());
            writer.write_u32(new_type);

            writer.write(tensor.extents.data(), tensor.extents.size());
            writer.write(tensor.name.data(), tensor.name.size());

            writer.seek(-writer.tell() & 31);

            FAST_LLAMA_ASSERT(new_size == tensor_size(tensor.extents, new_type), "tensor size mismatch");
            writer.write(new_data, new_size);

            return true;
        }

    };

    struct ModelLoader {
        bool                            is_load_failed{false};
        bool                            use_mmap{false};
        std::vector<FileLoader>         file_loaders;
        TensorsMapping                  tensors_map;
        std::size_t                     num_of_ggml_tensors_created{};
        MemContext                      mem_ctx;
        std::unique_ptr<MMappedFile>    mmapped_file{nullptr};
        Logger const*                   logger;

        ModelLoader(std::string_view fname_base, bool use_mmap, bool vocab_only, Logger const* logger) noexcept
            : use_mmap(use_mmap && MMappedFile::SUPPORTED)
            , logger(logger ? logger : &Logger::get_null_logger())
        {
            file_loaders.emplace_back(fname_base, 0ul, tensors_map, logger);
            auto& first_loader = file_loaders.back();

            if (first_loader.is_failed_to_read) {
                is_load_failed = true;
                return;
            }

            std::uint32_t num_of_files = vocab_only ? 1ul : guess_num_of_files();

            if (is_load_failed) return;

            for(auto i = 1ul; i < num_of_files; ++i) {
                std::string filename = std::string(fname_base) + "." + std::to_string(i);
                file_loaders.emplace_back(filename, i, tensors_map, logger);
                if (file_loaders.back().is_failed_to_read) {
                    is_load_failed = true;
                    return;
                }
                if (file_loaders.back().hyperparams != first_loader.hyperparams) {
                    logger->log_err(__func__, "Hyper parameters mismatch between '", filename, "' and '", fname_base, "'\n");
                    is_load_failed = true;
                    return;
                }
            }

            if (use_mmap && alignment_prevents_mmap()) {
                logger->log_warn(__func__, "can't use mmap because tensors are not aligned; convert to new format to avoid this\n");
                use_mmap = false;
            }

            this->use_mmap = use_mmap;
            for(auto& tensor : tensors_map.tensors) {
                if (!tensor.calc_all(*logger)) {
                    is_load_failed = true;
                    return;
                }
            }

        }

        auto alignment_prevents_mmap() const noexcept -> bool {
            for(auto const& tl : tensors_map.tensors) {
                for(auto const& shard : tl.shards) {
                    if (shard.file_off & 3) {
                        return true;
                    }
                }
            }
            return false;
        }

        auto guess_num_of_files() noexcept -> std::uint32_t {
            auto it = tensors_map.tensor_names.find("tok_embeddings.weight");
            if (it == tensors_map.tensor_names.end()) {
                logger->log_err(__func__, "tok_embeddings.weight not found\n");
                is_load_failed = true;
                return 0;
            }

            auto& tensor_loader = tensors_map.tensors[it->second];
            return file_loaders[0].hyperparams.n_embd / tensor_loader.shards[0].extents[0];
        }

        auto calc_sizes(std::size_t* ctx_size_p, std::size_t* mmapped_size_p) const noexcept -> bool {
            *ctx_size_p = 0;
            *mmapped_size_p = 0;
            for(auto const& tl : tensors_map.tensors) {
                *ctx_size_p += sizeof(ggml_tensor) + GGML_OBJECT_SIZE;
                *(use_mmap ? mmapped_size_p : ctx_size_p) += tl.size;
            }
            return true;
        }

        auto get_tensor(std::string_view name, std::vector<std::uint32_t> const& es) noexcept -> ggml_tensor* {
            auto it = tensors_map.tensor_names.find(name.data());

            if (it == tensors_map.tensor_names.end()) {
                logger->log_err(__func__, "tensor '", name, "' not found\n");
                return nullptr;
            }

            auto& tensor_loader = tensors_map.tensors[it->second];
            if (tensor_loader.extents != es) {
                logger->log_err(__func__, "tensor '", name, "' extents mismatch; expected ", format_tensor_shape(es),", got ", format_tensor_shape(tensor_loader.extents), "\n");
                return nullptr;
            }

            return get_tensor_for(tensor_loader);
        }

        auto get_tensor_for(TensorLoader& tl) noexcept -> ggml_tensor* {
            ggml_tensor* tensor{nullptr};
            if (tl.extents.size() == 2) {
                tensor = ggml_new_tensor_2d(mem_ctx.get(), tl.type, tl.extents[0], tl.extents[1]);
            } else {
                FAST_LLAMA_ASSERT(tl.extents.size() == 1, "invalid tensor rank");
                tensor = ggml_new_tensor_1d(mem_ctx.get(), tl.type, tl.extents[0]);
            }

            FAST_LLAMA_ASSERT(tensor, "failed to create tensor");
            tl.ggml_tensor = tensor;
            ++num_of_ggml_tensors_created;
            return tensor;
        }

        auto done_getting_tensors() const -> bool {
            if (num_of_ggml_tensors_created > tensors_map.tensors.size()) {
                logger->log_err(__func__, "file contained more tensors than expected\n");
                return false;
            }
            return num_of_ggml_tensors_created == tensors_map.tensors.size();
        }

        void load_all_data(MemoryLock* lmlock) {
            std::size_t data_size = 0;
            for(auto const& tl : tensors_map.tensors) {
                data_size += tl.size;
            }
            if (use_mmap) {
                mmapped_file.reset(new MMappedFile(&file_loaders[0].reader));
                if (!lmlock) {
                    // Don't call the callback since the actual loading will be lazy
                    // and we can't measure it.
                    call_progress_callback = false;
                } else {
                    lmlock->init(mmapped_file.get());
                }
            }

            std::size_t done_size{};
            for(auto& tl : tensors_map.tensors) {
                if (call_progress_callback) {
                    logger->progress(done_size, data_size);
                }

                FAST_LLAMA_ASSERT(tl.ggml_tensor, "tensor not created");
                tl.data = static_cast<std::uint8_t*>(tl.ggml_tensor->data);
                load_data_for(tl);
                tl.ggml_tensor->data = reinterpret_cast<void*>(tl.data);
                done_size += tl.size;

                if (use_mmap && lmlock) {
                    lmlock->grow_to(done_size);
                }
            }

            if (call_progress_callback) {
                logger->progress(data_size, data_size);
            }
        }

        void load_data_for(TensorLoader& tl) {
            if (use_mmap) {
                FAST_LLAMA_ASSERT(mmapped_file, "mmap file not created");
                tl.data = mmapped_file->get_data_offset(tl.shards[0].file_off);
            } else if (tl.split_type == SplitType::None) {
                auto& reader = file_loaders[tl.shards[0].file_idx].reader;
                reader.seek(tl.shards[0].file_off, BinaryFileReader::SeekReference::Begin);
                reader.read(tl.data, tl.size);
            } else if (tl.split_type == SplitType::ByRows) {
                std::size_t offset{};
                for(auto& shard : tl.shards) {
                    auto& reader = file_loaders[shard.file_idx].reader;
                    reader.seek(shard.file_off, BinaryFileReader::SeekReference::Begin);
                    reader.read(tl.data + offset, shard.size);
                    offset += shard.size;
                }
                FAST_LLAMA_ASSERT(offset == tl.size, "invalid tensor size");
            } else if (tl.split_type == SplitType::ByColumns) {
                // Let's load the data into temporary buffers to ensure the OS performs large loads.
                std::vector<UninitializedBuffer> tmp_buffers;
                tmp_buffers.reserve(tl.shards.size());
                for(auto& shard : tl.shards) {
                    auto& reader = file_loaders[shard.file_idx].reader;
                    reader.seek(shard.file_off, BinaryFileReader::SeekReference::Begin);
                    tmp_buffers.emplace_back(shard.size);
                    reader.read(tmp_buffers.back().data(), shard.size);
                }

                // Then reshape.
                std::size_t num_of_rows = tl.extents[1];
                std::size_t per_shard_row_size = tl.shards[0].size / num_of_rows;
                std::size_t out_offset{};

                for(auto i = 0ul; i < num_of_rows; ++i) {
                    for(auto& buff: tmp_buffers) {
                        std::memcpy(
                            tl.data + out_offset,
                            buff.data() + i * per_shard_row_size,
                            per_shard_row_size
                        );
                        out_offset += per_shard_row_size;
                    }
                }

                FAST_LLAMA_ASSERT(out_offset == tl.size, "invalid tensor size");
            }
        }

    private:
        bool call_progress_callback{true};

    };

} // namespace fastllama


#endif // FAST_LLAMA_FILE_LOADER_HPP
