#ifndef FAST_LLAMA_VOCAB_HPP
#define FAST_LLAMA_VOCAB_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>


namespace fastllama {
    
    struct Vocab {
        using id_type           = std::int32_t;
        using token_type        = std::string;
        using token_view_type   = std::string_view;
        
        auto get_token_from_id(id_type token_id) const noexcept -> std::string_view {
            auto temp_id = static_cast<std::size_t>(token_id);
            if (temp_id >= id_to_token.size()) return {};
            return id_to_token[temp_id].tok;
        }

        auto set_word(id_type token_id, std::string s, float score) {
            auto const temp_id = static_cast<std::size_t>(token_id);
            auto temp = std::string(s);
            id_to_token[temp_id] = {
                std::move(temp),
                score
            };
            token_to_id[id_to_token[temp_id].tok] = token_id;
        }

        struct token_score {
            token_type tok;
            float score;
        };

        std::unordered_map<token_view_type, id_type> token_to_id;
        std::vector<token_score> id_to_token;
    };
}

#endif // FAST_LLAMA_VOCAB_HPP