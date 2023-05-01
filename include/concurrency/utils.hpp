#if !defined(FAST_LLAMA_CONCURRENCY_UTILS_HPP)
#define FAST_LLAMA_CONCURRENCY_UTILS_HPP

#include "pool.hpp"
#include "macro.hpp"

namespace fastllama::parallel {
    
    struct Range {
        std::size_t start{};
        std::size_t end{};
        std::size_t block_size{1};
    };

    struct Block {
        std::size_t start{};
        std::size_t end{};
        std::size_t block_size{};
    };

    struct blocking_tag {};
    struct non_blocking_tag {};

    template<typename W>
    struct ForWorker{
        static_assert(std::is_invocable_v<W, Block>, "Type 'W' should invocable with two std::size_t arguments");
        W* work{nullptr};
        std::size_t i;
        std::size_t ib;

        void operator()() {
            std::invoke(*work, Block {
                i,
                i + ib,
                ib
            });
        }

        constexpr operator bool() const noexcept {
            return work != nullptr;
        }
    };

    namespace detail {
        
        template<typename T>
        struct is_for_woker : std::false_type {};

        template<typename W>
        struct is_for_woker<ForWorker<W>> : std::true_type {};

    } // namespace detail
    

    template<typename Tag = blocking_tag, typename W, typename V = ForWorker<W>>
    inline static auto for_(ThreadPool<V>& pool, Range range, W&& work, Tag tag = blocking_tag{}) {
        static_assert(std::is_invocable_v<W, Block>, "Type 'W' should invocable with two std::size_t arguments");
        FAST_LLAMA_ASSERT(range.start < range.end, "Range start should be less than range end");

        for(auto i = range.start; i < range.end; i += range.block_size) {
            auto ib = std::min(range.block_size, range.end - i);
            if constexpr(detail::is_for_woker<V>::value) {
                auto temp_work = ForWorker<W> {
                    &work,
                    i,
                    ib
                };
                pool.add_work(std::move(temp_work));
            } else {
                pool.add_work([&work, i, ib]() {
                    std::invoke(work, Block {
                        i,
                        i + ib,
                        ib
                    });
                });
            }
        }

        if constexpr (std::is_same_v<Tag, blocking_tag>) {
            pool.wait();
        }
    }

    // Always blocking
    template<typename InitialValue, typename Container, typename W, typename V>
    inline static auto reduce(ThreadPool<V>& pool, Container const& c, InitialValue value, W&& work, std::size_t block_size = 32, blocking_tag tag = blocking_tag{}) {
        static_assert(std::is_invocable_v<W, InitialValue, typename Container::value_type>, "Type 'W' should invocable with two arguments");

        auto result = value;
        auto range = Range{ 0, c.size(), block_size };
        auto size = (range.end / range.block_size) + 1;
        std::vector<InitialValue> temp_results(size, InitialValue{});
        auto id = 0ul;
        for(auto i = range.start; i < range.end; i += range.block_size, ++id) {
            auto ib = std::min(range.block_size, range.end - i);
            auto& temp_result = temp_results[id];
            pool.add_work([i, &work=work, ib, id, &c, &temp_result] {
                for(auto j = 0; j < ib; ++j) {
                    temp_result = std::invoke(work, temp_result, c[i + j]);
                }
            });
        }
        pool.wait();
        for(auto const& temp_result : temp_results) result = std::invoke(std::forward<W>(work), result, temp_result);
        return result;
    }

    template<typename Container, typename Tag = blocking_tag, typename W, typename V>
    inline static auto transform(ThreadPool<V>& pool, Container& c, W&& work, std::size_t block_size = 32, Tag tag = blocking_tag{}) {
        static_assert(std::is_invocable_v<W, typename Container::value_type>, "Type 'W' should invocable with one argument");

        auto range = Range{ 0, c.size(), block_size };

        for(auto i = range.start; i < range.end; i += range.block_size) {
            auto ib = std::min(range.block_size, range.end - i);
            pool.add_work([i, &work=work, ib, &c] {
                for(auto j = 0; j < ib; ++j) {
                    c[i + j] = std::invoke(work, c[i + j]);
                }
            });
        }

        if constexpr (std::is_same_v<Tag, blocking_tag>) {
            pool.wait();
        }
    }

} // namespace fastllama::parallel


#endif // FAST_LLAMA_CONCURRENCY_UTILS_HPP
