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
        using id    = std::int32_t;
        using token = std::string;

        constexpr auto get_token_from_id(id token_id) const noexcept -> std::string_view {
            if (token_id >= id_to_token.size()) return {};
            return id_to_token[token_id].tok;
        }

        struct token_score {
            token tok;
            float score;
        };

        std::unordered_map<token, id> token_to_id;
        std::vector<token_score> id_to_token;
    };
}

#endif // FAST_LLAMA_VOCAB_HPP