#if !defined(FAST_LLAMA_TOKEN_BUFFER_HPP)
#define FAST_LLAMA_TOKEN_BUFFER_HPP

#include "vocab.hpp"
#include <type_traits>
#include <deque>
#include <tuple>
#include <cassert>
#include <iostream>
#include "tokenizer.hpp"

namespace fastllama {
    
    struct TokenBufferPartialState {
        static constexpr std::size_t unicode_backlog_buffer_size = 8;

        std::string left_out_string{};
        std::size_t num_of_chars_in_unicode_buffer{0};
        char unicode_backlog_buffer[unicode_backlog_buffer_size] = {0};
    };

    template<typename Fn>
    struct TokenBuffer {
        using id_t = typename Vocab::id_type;
        static constexpr std::size_t str_buffer_size = 512;
        static constexpr std::size_t unicode_backlog_buffer_size = 8;

        TokenBuffer(Vocab const& vocab, std::size_t buffer_size, Fn&& fn)
            : m_vocab(vocab)
            , m_max_buffer_size(buffer_size)
            , m_fn(std::move(fn))
        {}

        auto add(id_t token_id) -> void {
            if (m_max_buffer_size <= m_buffer.size()) flush_buffer();
            m_buffer.push_back(token_id);
        }

        auto flush_buffer() -> void {
            if (m_buffer.empty()) return;
            auto const id = m_buffer.front();
            m_buffer.pop_front();

            auto temp = std::string(m_vocab.get_token_from_id(id));

            // Test if the last character is an invalid utf-8 character.
            // If an invalid Unicode is found, we remove it from the token
            // and wait for the other part to come, and then we prepend it to the next token.
            check_and_put_unicode_char_in_buffer_if_invalid(temp);
            if (temp.empty()) return;
            m_fn(std::move(temp));
        }

        constexpr auto are_tokens_present_in_buffer(Span<std::string> tokens) noexcept -> std::tuple<bool, std::string_view, std::string_view> {
            if (tokens.empty()) return std::make_tuple( false, std::string_view{}, std::string_view{} );

            assert(tokens.size() < (str_buffer_size - m_num_of_chars_in_unicode_buffer) && "Max token is reached");

            auto buffer_start = m_num_of_chars_in_unicode_buffer;
            
            // Copy any buffered unicode
            std::copy(m_unicode_backlog_buffer, m_unicode_backlog_buffer + m_num_of_chars_in_unicode_buffer, m_temp_str_buffer);

            std::for_each(m_buffer.begin(), m_buffer.end(), [&buffer_start, this](auto const e) {
                assert(buffer_start < (str_buffer_size - m_num_of_chars_in_unicode_buffer) && "Max token is reached");
                auto temp = m_vocab.get_token_from_id(e);
                std::copy(temp.begin(), temp.end(), m_temp_str_buffer + buffer_start);
                buffer_start += temp.size();
            });

            auto temp_str = std::string_view(m_temp_str_buffer, buffer_start);

            for (auto const& token : tokens) {
                auto substr_pos = temp_str.find(token);
                if (substr_pos != std::string_view::npos) {
                    return std::make_tuple(true, temp_str.substr(0, substr_pos), temp_str.substr(substr_pos + token.size()));
                }
            }

            return std::make_tuple( false, std::string_view{}, std::string_view{} );
        }

        void clear() {
            m_buffer.clear();
            m_num_of_chars_in_unicode_buffer = 0;
        }

        TokenBufferPartialState get_partial_state() const {
            auto temp = TokenBufferPartialState{};
            std::copy_n(m_unicode_backlog_buffer, m_num_of_chars_in_unicode_buffer, temp.unicode_backlog_buffer);
            temp.num_of_chars_in_unicode_buffer = m_num_of_chars_in_unicode_buffer;
            return temp;
        }

        void restore_partial_state(TokenBufferPartialState& state) {
            if (!state.left_out_string.empty()) {
                check_and_put_unicode_char_in_buffer_if_invalid(state.left_out_string);
                if (!state.left_out_string.empty()) {
                    m_fn(std::move(state.left_out_string));
                }
            }
            std::copy_n(state.unicode_backlog_buffer, state.num_of_chars_in_unicode_buffer, m_unicode_backlog_buffer);
            m_num_of_chars_in_unicode_buffer = state.num_of_chars_in_unicode_buffer;
        }

    private:
        
        void check_and_put_unicode_char_in_buffer_if_invalid(std::string& in_out) {
            if (in_out.empty()) return;

            // If backlog exists push it in front of the string
            if (m_num_of_chars_in_unicode_buffer != 0) {
                std::string temp(m_unicode_backlog_buffer, m_num_of_chars_in_unicode_buffer);
                in_out = temp + in_out;
                m_num_of_chars_in_unicode_buffer = 0;
            }

            auto str_len = in_out.size();
            std::size_t unicode_len{};
            std::size_t last_i = 0ul;
            
            // Assumption: Invalid unicode can occure at the last part and no other character exists after that
            for (std::size_t i{}; i < str_len;) {
                unicode_len = fastllama::utf8_len(in_out[i]);
                last_i = i;
                i += unicode_len;
            }

            if (last_i + unicode_len > str_len) {
                m_num_of_chars_in_unicode_buffer = str_len - last_i;
                std::copy(in_out.begin() + static_cast<std::ptrdiff_t>(last_i), in_out.end(), m_unicode_backlog_buffer);
                in_out.resize(last_i);
            }
        }

    private:
        Vocab const& m_vocab;
        std::deque<id_t> m_buffer;
        char m_temp_str_buffer[str_buffer_size];
        char m_unicode_backlog_buffer[unicode_backlog_buffer_size] = {0};
        std::size_t m_num_of_chars_in_unicode_buffer{0};
        std::size_t m_max_buffer_size{0};
        Fn m_fn;
    };

} // namespace fastllama


#endif // FAST_LLAMA_TOKEN_BUFFER_HPP
