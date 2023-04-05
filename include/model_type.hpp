#if !defined(FAST_LLAMA_MODEL_TYPE_HPP)
#define FAST_LLAMA_MODEL_TYPE_HPP

#include <array>
#include <string>
#include <limits>
#include <utility>
#include "utils.hpp"

namespace fastllama {

    struct ModelIdConfig {
        using size_type = std::size_t;

        size_type number_of_parts{};
        size_type mem_required_for_scratch_buff_0{};
        size_type mem_required_for_scratch_buff_1{};
        size_type mem_required_for_kv_self_buff{};
        size_type mem_required_for_eval{};
    };

    namespace detail {
        using namespace ::fastllama::literals;
        
        using model_lookup_table_t = std::pair<std::string_view, ModelIdConfig>;
        
        inline static constexpr model_lookup_table_t g_models[] = {
            model_lookup_table_t(
                "LLAMA-7B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 1,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 1026_MiB,
                    /* mem_required_for_eval =              */ 768_MiB
                }
            ),
            model_lookup_table_t(
                "LLAMA-13B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 2,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 1608_MiB,
                    /* mem_required_for_eval =              */ 1024_MiB
                }
            ),
            model_lookup_table_t(
                "LLAMA-30B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 4,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 3124_MiB,
                    /* mem_required_for_eval =              */ 1280_MiB
                }
            ),
            model_lookup_table_t(
                "LLAMA-65B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 8,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 5120_MiB,
                    /* mem_required_for_eval =              */ 1536_MiB
                }
            ),
            model_lookup_table_t(
                "ALPACA-LORA-7B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 1,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 1026_MiB,
                    /* mem_required_for_eval =              */ 768_MiB
                }
            ),
            model_lookup_table_t(
                "ALPACA-LORA-13B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 1,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 1608_MiB,
                    /* mem_required_for_eval =              */ 1024_MiB
                }
            ),
            model_lookup_table_t(
                "ALPACA-LORA-30B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 1,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 3124_MiB,
                    /* mem_required_for_eval =              */ 1280_MiB
                }
            ),
            model_lookup_table_t(
                "ALPACA-LORA-65B",
                ModelIdConfig {
                    /* number_of_parts =                    */ 1,
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 5120_MiB,
                    /* mem_required_for_eval =              */ 1536_MiB
                }
            ),
        };
    } // namespace detail
    


    struct ModelId;

    namespace detail {
        constexpr auto match_case_insensitive_str(std::string_view lhs, std::string_view rhs) noexcept -> bool {
            if (lhs.size() != rhs.size()) return false;
            for(auto i = 0ul; i < rhs.size(); ++i) {
                auto const lc = std::toupper(lhs[i]);
                auto const rc = std::toupper(rhs[i]);
                if (lc != rc) return false;
            }
            return true;
        }
    } // namespace detail

    struct ModelId {
        using id_t = std::string_view;

        id_t id{};
        ModelIdConfig config{};
        

        // Assumption 1: ModelId is a singleton
        // Assumption 2: It cannot be constructed outside this translation unit
        constexpr auto operator==(ModelId const& other) noexcept {
            return id.data() == other.id.data();
        }
        constexpr auto operator!=(ModelId const& other) noexcept {
            return !(*this == other);
        }

        constexpr auto operator==(std::string_view other) noexcept {
            return id == other;
        }

        constexpr auto operator!=(std::string_view other) noexcept {
            return !(*this == other);
        }

        constexpr operator bool() noexcept {
            return id.data() !=  nullptr && id.size() != 0;
        }

        constexpr static auto from_str_case_sensitive(std::string_view model_id) noexcept -> ModelId {
            for (auto const& model : detail::g_models) {
                auto const el = model.first;
                if (el == model_id) return ModelId{ model.first, model.second };
            }
            return ModelId{};
        }
        
        constexpr static auto from_str_case_insensitive(std::string_view model_id) noexcept -> ModelId {
            for (auto const& model : detail::g_models) {
                auto const el = model.first;
                if (detail::match_case_insensitive_str(el, model_id)) return ModelId{ model.first, model.second };
            }
            return ModelId{};
        }
    };

} // namespace fastllama


#endif // FAST_LLAMA_MODEL_TYPE_HPP

