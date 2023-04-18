#if !defined(FAST_LLAMA_DEQUE_HPP)
#define FAST_LLAMA_DEQUE_HPP

#include <type_traits>
#include <memory>
#include <cassert>
#include <atomic>
#include <vector>
#include <optional>

// This implementation is the work stealing queue described in the paper, 
// "Correct and Efficient Work-Stealing for Weak Memory Models,"
// available at https://www.di.ens.fr/~zappa/readings/ppopp13.pdf.

// This is implementation is inspired by https://github.com/ConorWilliams/ConcurrentDeque/blob/main/include/riften/deque.hpp

namespace fastllama{
    
    namespace detail {

        constexpr auto get_nearest_power_of_2(std::size_t size) noexcept -> std::size_t {
            if (size == 0) return 2ul;
            if (!(size & (size - 1))) return size;

            std::size_t exp{1};
            while(exp < size) {
                exp <<= 1;
            }
            return exp;
        }
        
        template<typename T>
        struct ArrayBuffer {
            static_assert(std::is_constructible_v<T>);
            using size_type = std::size_t;
            using base_type = std::unique_ptr<T[]>;

            ArrayBuffer(size_type capacity)
                : m_capacity{get_nearest_power_of_2(capacity)}
            {}

            ArrayBuffer(ArrayBuffer const&) = delete;
            ArrayBuffer(ArrayBuffer&&) noexcept = default;
            ArrayBuffer& operator=(ArrayBuffer const&) = delete;
            ArrayBuffer& operator=(ArrayBuffer&&) noexcept = default;
            ~ArrayBuffer() = default;

            constexpr auto capacity() const noexcept { return m_capacity; }

            constexpr auto store(size_type i, T&& data) noexcept {
                static_assert(std::is_nothrow_move_constructible_v<T>, "Move construction cannot throw");
                m_data[i & m_capacity] = std::move(data);
            }
            
            constexpr auto load(size_type i) noexcept -> T {
                static_assert(std::is_nothrow_move_constructible_v<T>, "Move construction cannot throw");
                return std::move(m_data[i & m_capacity]);
            }

            auto resize(size_type b, size_type t) -> ArrayBuffer* {
                ArrayBuffer* ptr = new ArrayBuffer(m_capacity << 1);
                for(auto i = t; i != b; ++i){
                    ptr->store(i, load(i));
                }
                return ptr;
            }
        private:
            size_type m_capacity;
            size_type m_mask{m_capacity - 1};
            base_type m_data { new T[m_capacity] };
        };

        #ifdef __cpp_lib_hardware_interference_size
            using std::hardware_destructive_interference_size;
        #else
            // 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
            inline static constexpr std::size_t hardware_destructive_interference_size = 2 * sizeof(std::max_align_t);
        #endif

        template<typename T>
        void atomic_swap(std::atomic<T>& a, std::atomic<T>& b) noexcept {
            auto val = a.load();
            a.exchange(b.load());
            b.exchange(val);
        }

    } // namespace detail
    
    template<typename T>
    struct Deque {
        using base_type = detail::ArrayBuffer<T>;
        using size_type = typename base_type::size_type;
        using garbage_value_type = std::unique_ptr<base_type>;

        Deque(size_type capacity = 1024ul)
            : m_top(0)
            , m_bottom{0}
            , m_data(new base_type(capacity))
        {
            m_garbage.reserve(32);
        }

        Deque(Deque const&) = delete;
        Deque(Deque&& other) noexcept {
            swap(*this, other);
        }
        Deque& operator=(Deque const&) = delete;
        Deque& operator=(Deque&& other) noexcept {
            auto temp = Deque(std::move(other));
            swap(*this, temp);
            return *this;
        }
        ~Deque() { delete m_data.load(); }

        constexpr auto size() const noexcept -> size_type {
            auto b = m_bottom.load(std::memory_order_relaxed);
            auto t = m_top.load(std::memory_order_relaxed);
            return b > t ? b - t : 0;
        }

        constexpr auto capacity() const noexcept -> size_type {
            return m_data.load(std::memory_order_relaxed)->capacity();
        }

        constexpr auto empty() const noexcept -> bool {
            return size() == 0;
        }

        template<typename... Args>
        void emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
            T obj(std::forward<Args>(args)...);
            auto b = m_bottom.load(std::memory_order_relaxed);
            auto t = m_top.load(std::memory_order_acquire);

            auto next_b = b + 1;

            base_type* buf = m_data.load(std::memory_order_relaxed);
            auto size = 1 + (b >= t ? b - t : 0);
            if (buf->capacity() < size) {
                m_garbage.emplace_back(std::exchange(buf, buf->resize(b, t)));
                m_data.store(buf, std::memory_order_relaxed);
            }

            buf->store(b, std::move(obj));
            std::atomic_thread_fence(std::memory_order_relaxed);
            m_bottom.store(next_b, std::memory_order_relaxed);
        }

        std::optional<T> pop() noexcept {
            auto const temp_b = m_bottom.load(std::memory_order_relaxed);
            if (temp_b == 0) return {};
            auto const b = temp_b - 1ul;
            base_type* buf = m_data.load(std::memory_order_relaxed);
            m_bottom.store(b, std::memory_order_relaxed);

            std::atomic_thread_fence(std::memory_order_seq_cst);

            auto t = m_top.load(std::memory_order_relaxed);

            if (t > b) {
                m_bottom.store(b + 1, std::memory_order_relaxed);
                return {};
            }

            if (t == b) {
                if (!m_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    m_bottom.store(b + 1, std::memory_order_relaxed);
                    return {};
                }

                m_bottom.store(b + 1, std::memory_order_relaxed);
            }

            return buf->load(b);
        }

        std::optional<T> steal() noexcept {
            auto t = m_top.load(std::memory_order_acquire);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            auto const b = m_bottom.load(std::memory_order_acquire);

            if (t >= b) return {};

            // Must load *before* acquiring the slot as slot may be overwritten immediately after acquiring.
            // This load is NOT required to be atomic even-though it may race with an overwrite as we only
            // return the value if we win the race below, it guaranties that we had no race during our read. If we
            // loose the race then 'x' could be corrupt due to read-during-write race but as T is trivially
            // destructible this does not matter.
            T obj = m_data.load(std::memory_order_consume)->load(t);

            if (!m_top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return {};
            }

            return std::move(obj);
        }

        friend void swap(Deque& lhs, Deque& rhs) {
            detail::atomic_swap(lhs.m_bottom, rhs.m_bottom);
            detail::atomic_swap(lhs.m_top, rhs.m_top);
            detail::atomic_swap(lhs.m_data, rhs.m_data);
            std::swap(lhs.m_garbage, rhs.m_garbage);
        }

    private:
        alignas(detail::hardware_destructive_interference_size) std::atomic<size_type> m_top;
        alignas(detail::hardware_destructive_interference_size) std::atomic<size_type> m_bottom;
        alignas(detail::hardware_destructive_interference_size) std::atomic<base_type*> m_data;

        std::vector<garbage_value_type> m_garbage;
    };

} // namespace fastllama


#endif // FAST_LLAMA_DEQUE_HPP
