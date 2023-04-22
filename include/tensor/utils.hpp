#if !defined(FAST_LLAMA_TENSOR_UTILS_HPP)
#define FAST_LLAMA_TENSOR_UTILS_HPP

#include <vector>
#include "logger.hpp"
#include "maths_utils.hpp"
#include "ggml.h"

namespace fastllama {

    struct HyperParams;
    
    inline static constexpr std::optional<std::size_t> tensor_size(std::vector<std::uint32_t> const& shape, enum ggml_type type) noexcept {
        std::size_t size = 1;
        for (auto s : shape) {
            auto new_size = checked_mul(size, static_cast<std::size_t>(s));
            if (!new_size) return {};
            size = new_size.value();
        }
        return size / ggml_blck_size(type);
    }

    inline static std::string format_tensor_shape(std::vector<std::uint32_t> const& shape) {
        char buffer[256];
        auto offset = format_str(buffer, "%5u", shape[0]).size();
        for(auto i = 1ul; i < shape.size(); ++i) {
            offset += format_str(buffer + offset, 256, "x %5u", shape[i]).size();
        }
        return std::string(buffer, offset);
    }

    struct TensorShard {
        std::vector<std::uint32_t>  extents;
        std::size_t                 size;
        enum ggml_type              type;
        std::size_t                 file_idx;
        std::size_t                 file_off;

        constexpr bool calc_size(Logger const& logger) noexcept {
            auto size = tensor_size(extents, type);
            if (!size) {
                logger.log_err(__func__, "Failed to calculate tensor size. Overflow detected while multiplying extents.\n");
                return false;
            }
            this->size = *size;
            return true;
        }
    };

    enum class SplitType {
        None,
        ByColumns,
        ByRows
    };

    struct TensorLoader {
        std::vector<TensorShard>   shards;
        std::string                 name;
        enum ggml_type              type{GGML_TYPE_F32};
        SplitType                   split_type{SplitType::None};
        std::vector<std::uint32_t>  extents;
        std::size_t                 size{0};
        ggml_tensor*                ggml_tensor{nullptr};
        std::uint8_t*               data{nullptr};

        TensorLoader(std::string_view name)
            : name(name)
        {}

        TensorLoader(TensorLoader const&) = delete;
        TensorLoader(TensorLoader&&) = default;
        TensorLoader& operator=(TensorLoader const&) = delete;
        TensorLoader& operator=(TensorLoader&&) = default;
        ~TensorLoader() = default;


        bool calc_type(Logger const& logger) noexcept {
            if (shards.empty()) {
                logger.log_err(__func__, "No shards found for tensor '", name, "'\n");
                return false;
            }
            auto const& first_shard = shards[0];
            for(auto const& shard : shards) {
                if (shard.type != first_shard.type) {
                    logger.log_err(__func__, "inconsistent tensor shard type in, '", name,"'\n");
                    return false;
                }
            }
            type = first_shard.type;
            return true;
        }

        bool calc_split_type(Logger const& logger) noexcept {
            if (shards.empty()) {
                logger.log_err(__func__, "No shards found for tensor '", name, "'\n");
                return false;
            }
            
            if (shards[0].extents.size() != 1 || shards.size() == 1 /* Maybe has only one file */) {
                split_type = SplitType::None;
            } else if (
                name.find("tok_embeddings.") == 0                       ||
                name.find(".attention.wo.weight") != std::string::npos  ||
                name.find(".feed_forward.w2.weight") != std::string::npos
            ) {
                split_type = SplitType::ByColumns;
            } else {
                split_type = SplitType::ByRows;
            }
            
            return true;
        }

        bool calc_extents(Logger const& logger) noexcept {
            if (shards.empty()) {
                logger.log_err(__func__, "No shards found for tensor '", name, "'\n");
                return false;
            }
            auto const& first_shard = shards[0];
            for(auto const& shard : shards) {
                if (shard.extents != first_shard.extents) {
                    logger.log_err(__func__, "inconsistent tensor shard extents in, '", name,"'\n");
                    return false;
                }
            }
            extents = first_shard.extents;

            auto shards_len = static_cast<std::uint32_t>(shards.size());
            if (split_type == SplitType::ByColumns) {
                auto e0 = checked_mul<std::uint32_t>(first_shard.extents[0], shards_len);
                if (!e0) {
                    logger.log_err(__func__, "Failed to calculate tensor extents for split type 'ByColumns'. Overflow detected while multiplying extent with number of shards.\n");
                    return false;
                }
                auto e1 = first_shard.extents[1];
                extents = { e0.value(), e1 };
            } else if (split_type == SplitType::ByRows) {
                auto e0 = first_shard.extents[0];
                auto e1 = checked_mul<std::uint32_t>(first_shard.extents[1], shards_len);
                if (!e1) {
                    logger.log_err(__func__, "Failed to calculate tensor extents for split type 'ByRows'. Overflow detected while multiplying extent with number of shards.\n");
                    return false;
                }
                extents = { e0, e1.value() };
            }
            return true;
        }

        bool calc_size(Logger const& logger) noexcept {
            auto size = tensor_size(extents, type);
            if (!size) {
                logger.log_err(__func__, "Failed to calculate tensor size. Overflow detected while multiplying extents.\n");
                return false;
            }
            this->size = *size;
            return true;
        }

        bool calc_all(Logger const& logger) noexcept {
            return  calc_type(logger)       &&
                    calc_split_type(logger) &&
                    calc_extents(logger)    &&
                    calc_size(logger);
        }
    };

    struct TensorsMapping {
        std::vector<TensorLoader> tensors;
        std::unordered_map<std::string, std::size_t> tensor_names;

        auto operator[](std::size_t k) const noexcept -> ggml_tensor* {
            return tensors[k].ggml_tensor;
        }

        auto operator[](std::string_view k) const noexcept -> ggml_tensor* {
            auto it = tensor_names.find(std::string(k));
            if (it == tensor_names.end()) return nullptr;
            return tensors[it->second].ggml_tensor;
        }

        auto operator[](std::string const& k) -> ggml_tensor*& {
            auto it = tensor_names.find(k);
            if (it == tensor_names.end()) {
                tensors.emplace_back(nullptr);
                tensor_names[k] = tensors.size() - 1;
                return tensors.back().ggml_tensor;
            };
            return tensors[it->second].ggml_tensor;
        }

        auto contains(char const* data) const noexcept -> bool {
            return tensor_names.find(data) != tensor_names.end();
        }
        
        auto contains(std::string const& data) const noexcept -> bool {
            return tensor_names.find(data) != tensor_names.end();
        }

        bool initializeTensors(ggml_context* ctx, HyperParams const& params, Logger const& logger = Logger{});

        auto make_tensors_by_name() -> std::unordered_map<std::string, ggml_tensor*> {
            std::unordered_map<std::string, ggml_tensor*> ret;
            for (auto const& [k, v] : tensor_names) {
                ret[k] = tensors[v].ggml_tensor;
            }
            return ret;
        }
    };

} // namespace fastllama


#endif // FAST_LLAMA_TENSOR_UTILS_HPP
