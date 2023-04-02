#ifndef FAST_LLAMA_TOKENIZER_HPP
#define FAST_LLAMA_TOKENIZER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include "vocab.hpp"


// Original implemnetation: https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp
namespace fastllama {
    static constexpr std::size_t const utf_8_lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };

    constexpr auto utf8_len(char src) noexcept -> std::size_t {
        auto const highbits = static_cast<std::uint8_t>(src) >> 4;
        return utf_8_lookup[highbits];
    }

    constexpr auto combine_char_helper(char c, uint8_t shift) noexcept {
        return (static_cast<int32_t>(c) & 0xff) << shift;
    }

    constexpr auto combine_char(char const* data, std::size_t len) -> int32_t {
        auto temp = int32_t{};
        switch (len) {
            case 2:
                temp = combine_char_helper(data[0], 8) | combine_char_helper(data[1], 0);
                break;
            case 3:
                temp = combine_char_helper(data[0], 16) | combine_char_helper(data[1], 8) | combine_char_helper(data[2], 0);
                break;
            case 4:
            // std::cout<<"["<<combine_char_helper(data[0], 0)<< ", " << combine_char_helper(data[1], 0)<< ", "<<combine_char_helper(data[2], 0)<<", "<<combine_char_helper(data[3], 0) << "]";
                temp = combine_char_helper(data[0], 24) | combine_char_helper(data[1], 16) | combine_char_helper(data[2], 8) | combine_char_helper(data[3], 0);
                break;
            default: temp = data[0];
        }
        return temp;
    }

    struct sp_symbol {
        using index_t = int;
        std::string_view text;
        index_t prev{-1};
        index_t next{-1};

        constexpr auto clear() noexcept {
            text = std::string_view{};
        }

        constexpr auto merge_symbol(sp_symbol const& other) noexcept {
            text = std::string_view(text.data(), text.size() + other.text.size());
            next = other.next;
        }
    };

    struct sp_bigram {
        struct comparator {
            constexpr auto operator()(sp_bigram & l, sp_bigram & r) const noexcept -> bool {
                return (l.score < r.score) || (l.score == r.score && l.left > r.left);
            }
        };
        using queue_storage_t = std::vector<sp_bigram>;
        using queue_t = std::priority_queue<sp_bigram, queue_storage_t, comparator>;
        typename sp_symbol::index_t left;
        typename sp_symbol::index_t right;
        float score;
        size_t size;
    };

    struct tokenizer {
        using index_t = typename sp_symbol::index_t;
        using queue_t = typename sp_bigram::queue_t;

        tokenizer(Vocab const& v)
            : m_vocab(v)
        {}

        auto operator()(std::string_view text, std::vector<typename Vocab::id>& out) {
            auto index = 0l;
            auto offset = std::size_t{};
            while(offset < text.size()) {
                auto sym = sp_symbol{};
                auto char_len = std::min(text.size() - offset, utf8_len(text[offset]));
                sym.text = std::string_view( text.data() + offset, char_len );
                offset += char_len;
                sym.prev = index - 1;
                sym.next = (offset == text.size()) ? -1 : index + 1;
                ++index;
                m_symbols.push_back(sym);
            }

            for(auto i = 1ul; i < m_symbols.size(); ++i) {
                try_add_bigram(static_cast<index_t>(i - 1), static_cast<index_t>(i));
            }

            while(!m_queue.empty()) {
                auto const bigram = m_queue.top();
                m_queue.pop();

                auto& left_sym = m_symbols[bigram.left];
                auto& right_sym = m_symbols[bigram.right];

                auto const sym_size = left_sym.text.size() + right_sym.text.size();
                if (left_sym.text.empty() || right_sym.text.empty() || sym_size != bigram.size) {
                    continue;
                }

                left_sym.merge_symbol(right_sym);

                if (right_sym.next >= 0) {
                    m_symbols[right_sym.next].prev = bigram.left;
                }

                right_sym.clear();

                try_add_bigram(left_sym.prev, bigram.left);
                try_add_bigram(bigram.left, left_sym.next);
            }

            for(index_t i = 0; i != -1; i = m_symbols[i].next) {
                auto& sym = m_symbols[i];

                if (auto const& token = m_vocab.token_to_id.find(std::string(sym.text)); token != m_vocab.token_to_id.end()) {
                    out.push_back(token->second);
                } else {
                    for(auto const c : sym.text) {
                        auto id = (static_cast<typename Vocab::id>(c) & 0xff);
                        out.push_back(id + 3);
                    }
                }
            }
        }

    private:
        auto try_add_bigram(index_t left, index_t right) -> void {
            if (left == -1 || right == -1) return;

            auto const text = std::string(m_symbols[left].text.data(), m_symbols[left].text.size() + m_symbols[right].text.size());
            auto const token = m_vocab.token_to_id.find(text);
            if (token == m_vocab.token_to_id.end()) return;

            auto const token_id = static_cast<std::size_t>(token->second);
            if (token_id >= m_vocab.id_to_token.size()) return;

            auto const& tok_score = m_vocab.id_to_token[token_id];

            sp_bigram bigram;
            bigram.left = left;
            bigram.right = right;
            bigram.score = tok_score.score;
            bigram.size = text.size();
            m_queue.push(bigram);
        }

    private:
        Vocab const& m_vocab;
        std::vector<sp_symbol> m_symbols;
        queue_t m_queue;
    };

    auto tokenize(Vocab const& v, std::string_view text, bool bos) {
        auto tok = tokenizer(v);
        std::vector<typename Vocab::id> out;
        
        if (text.size() == 0) return out;
        if (bos) out.push_back(1);

        tok(text, out);

        return out;
    }
}

#endif // FAST_LLAMA_TOKENIZER_HPP