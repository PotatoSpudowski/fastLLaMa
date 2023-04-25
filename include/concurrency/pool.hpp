#if !defined(FAST_LLAMA_POOL_HPP)
#define FAST_LLAMA_POOL_HPP

#include <thread>
#include "deque.hpp"
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>

using namespace std::chrono_literals;

namespace fastllama {

    struct ThreadPool {

        using WorkerFn = std::function<void()>;

        ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
            : m_num_threads(num_threads)
            , m_stop{false}
            , m_pending_tasks{0}
        {}

        void start() {
            m_threads.reserve(m_num_threads);
            m_worker_tasks.reserve(m_num_threads);
            for (auto i = 0ul; i < m_num_threads; ++i) {
                m_worker_tasks.emplace_back();
                m_threads.emplace_back([this, i] { work(i); });
            }
        }

        void stop() noexcept {
            m_stop.store(true, std::memory_order_relaxed);
        }

        bool is_stopped() const noexcept {
            return m_stop.load(std::memory_order_relaxed);
        }

        void wait() noexcept {
            std::mutex mtx;
            while(m_pending_tasks.load(std::memory_order_relaxed) > 0) {
                std::unique_lock<std::mutex> lock{mtx};
                m_wait_cv.wait_for(lock, 1ms, [this] { return m_pending_tasks.load(std::memory_order_relaxed) == 0; });
            }
        }

        void add_work(WorkerFn&& fn) {
            if (m_threads.empty() || m_num_threads == 1) {
                fn();
                return;
            }
            auto& task_queue = m_worker_tasks[m_current_worker];
            task_queue.emplace(std::move(fn));
            m_pending_tasks.fetch_add(1, std::memory_order_relaxed);
            m_cv.notify_all();
            m_current_worker = (m_current_worker + 1) % m_num_threads;
        }

        ~ThreadPool() {
            join_threads();
        }

    private:

        void join_threads() {
            stop();
            m_cv.notify_all();
            for (auto& thread : m_threads) {
                if (thread.joinable()) thread.join();
            }
        }

        void work(std::size_t id) {
            auto& task_queue = m_worker_tasks[id];
            while (!m_stop) {
                while(!task_queue.empty() && !m_stop) {
                    auto task = task_queue.pop();
                    if (task.has_value() && task.value()) {
                        (*task)();
                        m_pending_tasks.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
                if (task_queue.empty() && try_steal(id)) continue;
                std::unique_lock<std::mutex> lock{m_mtx};
                m_cv.wait(lock, [this, &task_queue] { return m_stop || !task_queue.empty(); });
                // std::printf("Worker %lu woke up with %lu tasks\n", id, task_queue.size());
            }
        }

        bool try_steal(std::size_t id) {
            for (auto i = 0ul; i < m_worker_tasks.size(); ++i) {
                if (i == id) continue;
                auto& task_queue = m_worker_tasks[i];
                if (task_queue.empty()) continue;
                auto task = task_queue.steal();
                if (task.has_value()) {
                    m_worker_tasks[id].emplace(std::move(*task));
                    return true;
                }
            }
            return false;
        }
        
    private:
        std::size_t                                                                             m_num_threads;
        std::vector<std::thread>                                                                m_threads;
        std::vector<Deque<WorkerFn>>                                                            m_worker_tasks;
        alignas(detail::hardware_destructive_interference_size) std::atomic<bool>               m_stop;
        alignas(detail::hardware_destructive_interference_size) std::atomic<std::size_t>        m_pending_tasks;
        std::mutex                                                                              m_mtx;
        std::condition_variable                                                                 m_cv;
        std::size_t                                                                             m_current_worker{0};
        std::condition_variable                                                                 m_wait_cv;
    };

} // namespace fastllama


#endif // FAST_LLAMA_POOL_HPP
