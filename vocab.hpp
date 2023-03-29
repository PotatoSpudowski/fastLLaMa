#ifndef FAST_LLAMA_VOCAB_HPP
#define FAST_LLAMA_VOCAB_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>


namespace fastllama {
    
    struct vocab {
        using id    = int32_t;
        using token = std::string;

        struct token_score {
            token tok;
            float score;
        };

        std::unordered_map<token, id> token_to_id;
        std::vector<token_score> id_to_token;
    };
}

#endif // FAST_LLAMA_VOCAB_HPP