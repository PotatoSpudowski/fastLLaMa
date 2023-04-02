#if !defined(FAST_LLAMA_MODEL_TYPE_HPP)
#define FAST_LLAMA_MODEL_TYPE_HPP

#include <array>
#include <string>
#include <limits>
#include <utility>

namespace fastllama {

    inline static constexpr std::pair<std::string_view const, std::size_t> g_models[] = {
        std::pair( "LLAMA-7B", 1 ),
        std::pair( "LLAMA-13B", 2 ),
        std::pair( "LLAMA-30B", 4 ),
        std::pair( "LLAMA-65B", 8 ),
        std::pair( "ALPACA-LORA-7B", 1 ),
        std::pair( "ALPACA-LORA-13B", 1 ),
        std::pair( "ALPACA-LORA-30B", 1 ),
        std::pair( "ALPACA-LORA-65B", 1 ),
    };

    struct ModelId;

    namespace detail {
        constexpr auto match_case_insenstive_str(std::string_view lhs, std::string_view rhs) noexcept -> bool {
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
        using part_t = std::size_t;

        id_t id{};
        part_t number_of_parts{0};

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

        constexpr static auto from_str_case_senstive(std::string_view model_id) noexcept -> ModelId {
            for (auto const& model : g_models) {
                auto const el = model.first;
                if (el == model_id) return ModelId{ model.first, model.second };
            }
            return ModelId{};
        }
        
        constexpr static auto from_str_case_insenstive(std::string_view model_id) noexcept -> ModelId {
            for (auto const& model : g_models) {
                auto const el = model.first;
                if (detail::match_case_insenstive_str(el, model_id)) return ModelId{ model.first, model.second };
            }
            return ModelId{};
        }
    };

} // namespace fastllama


#endif // FAST_LLAMA_MODEL_TYPE_HPP

