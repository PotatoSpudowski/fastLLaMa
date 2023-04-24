#if !defined(FAST_LLAMA_MODEL_TYPE_HPP)
#define FAST_LLAMA_MODEL_TYPE_HPP

#include <array>
#include <string>
#include <limits>
#include <utility>
#include "utils.hpp"
#include "macro.hpp"
#include <iostream>

namespace fastllama {

    struct ModelIdConfig {
        using size_type = std::size_t;

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
                "7B",
                ModelIdConfig {
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 1026_MiB,
                    /* mem_required_for_eval =              */ 768_MiB,
                }
            ),
            model_lookup_table_t(
                "13B",
                ModelIdConfig {
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 1608_MiB,
                    /* mem_required_for_eval =              */ 1024_MiB,
                }
            ),
            model_lookup_table_t(
                "30B",
                ModelIdConfig {
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 3124_MiB,
                    /* mem_required_for_eval =              */ 1280_MiB,
                }
            ),
            model_lookup_table_t(
                "65B",
                ModelIdConfig {
                    /* mem_required_for_scratch_buff_0 =    */ 512_MiB,
                    /* mem_required_for_scratch_buff_1 =    */ 512_MiB,
                    /* mem_required_for_kv_self_buff =      */ 5120_MiB,
                    /* mem_required_for_eval =              */ 1536_MiB,
                }
            ),
        };
        inline static constexpr auto g_models_size_v = sizeof(detail::g_models) / sizeof(detail::g_models[0]);
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

        template<std::size_t N>
        constexpr auto get_model_index(char const name[N]) noexcept -> std::size_t {
            std::size_t i = 0ul;
            auto temp_name = std::string_view(name);
            for(auto const& m : g_models) {
                if (m.first == temp_name) return i;
                ++i;
            }
            return i;
        }

    } // namespace detail

    #define GET_MODE_IDX(STR) ([]() { constexpr auto index = detail::get_model_index<sizeof(STR)>(STR);  static_assert(index < detail::g_models_size_v, "Model(='" STR "') does not exist."); return index; })()

    enum class ModelKind : std::size_t {
        Model_7B = GET_MODE_IDX("7B"),
        Model_13B = GET_MODE_IDX("13B"),
        Model_30B = GET_MODE_IDX("30B"),
        Model_65B = GET_MODE_IDX("65B")
    };

    constexpr std::string_view to_string_view(ModelKind const model) noexcept {
        switch(model) {
            case ModelKind::Model_7B: return "7B";
            case ModelKind::Model_13B: return "13B";
            case ModelKind::Model_30B: return "30B";
            case ModelKind::Model_65B: return "65B";
        }
        FAST_LLAMA_ASSERT(false, "Invalid model kind");
    }

    #undef GET_MODE_IDX

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

