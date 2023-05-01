#ifndef FAST_LLAMA_LOCKED_QUEUE_HPP
#define FAST_LLAMA_LOCKED_QUEUE_HPP

#include <type_traits>
#include <memory>
#include <cassert>
#include <atomic>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace fastllama {

    template<typename T>
    class LockedQueue {
    public:
        using size_type = std::size_t;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using iterator = typename std::deque<T>::iterator;
        using const_iterator = typename std::deque<T>::const_iterator;
        using reverse_iterator = typename std::deque<T>::reverse_iterator;
        using const_reverse_iterator = typename std::deque<T>::const_reverse_iterator;

        LockedQueue() = default;
        LockedQueue(const LockedQueue& other)
            : m_queue(other.m_queue)
        {}
        LockedQueue(LockedQueue&& other) noexcept
            : m_queue(std::move(other.m_queue))
        {}
        LockedQueue& operator=(const LockedQueue& other) {
            if (this != &other) {
                std::scoped_lock lock(m_mutex, other.m_mutex);
                m_queue = other.m_queue;
            }
            return *this;
        }
        LockedQueue& operator=(LockedQueue&& other) noexcept {
            if (this != &other) {
                std::scoped_lock lock(m_mutex, other.m_mutex);
                m_queue = std::move(other.m_queue);
            }
            return *this;
        }
        ~LockedQueue() = default;

        bool empty() const noexcept {
            std::scoped_lock lock(m_mutex);
            return m_queue.empty();
        }
        
        bool weak_empty() const noexcept {
            return m_queue.empty();
        }

        size_type size() const noexcept {
            std::scoped_lock lock(m_mutex);
            return m_queue.size();
        }

        const_reference front() const {
            std::scoped_lock lock(m_mutex);
            return m_queue.front();
        }

        reference front() {
            std::scoped_lock lock(m_mutex);
            return m_queue.front();
        }

        const_reference back() const {
            std::scoped_lock lock(m_mutex);
            return m_queue.back();
        }

        reference back() {
            std::scoped_lock lock(m_mutex);
            return m_queue.back();
        }

        void push(const T& value) {
            std::scoped_lock lock(m_mutex);
            m_queue.push_back(value);
        }

        void push(T&& value) {
            std::scoped_lock lock(m_mutex);
            m_queue.push_back(std::move(value));
        }

        template<typename... Args>
        void emplace(Args&&... args) {
            std::scoped_lock lock(m_mutex);
            m_queue.emplace_back(std::forward<Args>(args)...);
        }

        std::optional<T> pop() {
            std::unique_lock lock(m_mutex, std::defer_lock);
            if (!lock.try_lock()) return std::nullopt;
            if (m_queue.empty()) return std::nullopt;
            T value = std::move(m_queue.front());
            m_queue.pop_front();
            return std::move(value);
        }

        std::optional<T> steal() {
            return pop();
        }

    private:
        std::deque<T> m_queue;
        mutable std::mutex m_mutex{};
    };

} // namespace fastllama

#endif // FAST_LLAMA_LOCKED_QUEUE_HPP
