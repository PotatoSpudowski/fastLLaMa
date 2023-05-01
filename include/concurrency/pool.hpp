#if !defined(FAST_LLAMA_POOL_HPP)
#define FAST_LLAMA_POOL_HPP

#include <thread>
#include "deque.hpp"
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <optional>
#include <random>
#include <algorithm>
#include "lock_queue.hpp"

// #if defined(_WIN32)
//     #include <windows.h>
// #elif defined(__linux__)
//     #include <pthread.h>
//     #include <sched.h>
// #endif

// #if defined(__APPLE__) && defined(__MACH__)
//     #include <pthread.h>
//     #include <dispatch/dispatch.h>
// #endif

namespace fastllama {
    namespace detail {
        // enum class QOSClass {
        //     USER_INTERACTIVE = QOS_CLASS_USER_INTERACTIVE,
        //     USER_INITIATED = QOS_CLASS_USER_INITIATED,
        //     DEFAULT = QOS_CLASS_DEFAULT,
        //     UTILITY = QOS_CLASS_UTILITY,
        //     BACKGROUND = QOS_CLASS_BACKGROUND,
        // };

        // inline static void pin_thread_to_core(std::size_t core_id) {
        //     #if defined(_WIN32)
        //         SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << core_id);
        //     #elif defined(__linux__)
        //         cpu_set_t cpuset;
        //         CPU_ZERO(&cpuset);
        //         CPU_SET(core_id, &cpuset);
        //         pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        //     #endif
        // }

        // inline static bool set_qos_class_for_current_thread(QOSClass qos_class, int priority) {
        //     int result = pthread_set_qos_class_self_np(static_cast<qos_class_t>(qos_class), priority);
        //     return result == 0;
        // }
    }

    template<typename WorkerFn = std::function<void()>>
    struct ThreadPool {

        static_assert(std::is_invocable_r_v<void, WorkerFn>, "WorkerFn must be invocable with no arguments and return void");

        ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
            : m_num_threads(num_threads)
            , m_stop{false}
            , m_pending_tasks{0}
        {}

        void start() {
            if (m_threads.size() > 0) return;
            m_threads.reserve(m_num_threads);
            m_worker_tasks.reserve(m_num_threads);
            
            auto temp_mutexes = std::vector<std::mutex>(m_num_threads);
            std::swap(temp_mutexes, m_worker_mutexes);

            auto temp_cvs = std::vector<std::condition_variable>(m_num_threads);
            std::swap(temp_cvs, m_cvs);

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

        void wait() {
            while(m_pending_tasks.load(std::memory_order_relaxed) > 0) {
                std::unique_lock<std::mutex> lock{m_wait_mtx};
                m_wait_cv.wait_for(lock, std::chrono::microseconds(1), [this] { return m_pending_tasks.load(std::memory_order_relaxed) == 0; });
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
            m_cvs[m_current_worker].notify_all();
            m_current_worker = (m_current_worker + 1) % m_worker_tasks.size();
        }

        ~ThreadPool() {
            join_threads();
        }

    private:

        void join_threads() {
            stop();
            for(auto& cv : m_cvs) cv.notify_all();
            for (auto& thread : m_threads) {
                if (thread.joinable()) thread.join();
            }
        }

        void work(std::size_t id) {
            // detail::pin_thread_to_core(id);
            // detail::set_qos_class_for_current_thread(detail::QOSClass::USER_INITIATED, 0);
            auto& task_queue = m_worker_tasks[id];
            auto& mutex = m_worker_mutexes[id];
            auto& cv = m_cvs[id];

            while (!m_stop) {
                std::optional<WorkerFn> task;
                while(!m_stop.load(std::memory_order_relaxed) && (task = task_queue.pop()).has_value()) {
                    if (task.value()) {
                        std::invoke(task.value());
                        m_pending_tasks.fetch_sub(1, std::memory_order_relaxed);
                    } else break;
                }
                if (task_queue.weak_empty()) {
                    if (try_steal(id)) continue;
                    else std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                std::unique_lock<std::mutex> lock{mutex};
                cv.wait(lock, [this, &task_queue] { return m_stop || !task_queue.empty(); });
            }
        }

        bool try_steal(std::size_t id) {
            thread_local std::default_random_engine random_engine{std::random_device{}()};
            thread_local std::vector<std::size_t> indices(m_num_threads);
            
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), random_engine);

            for (auto i : indices) {
                if (i == id) continue;
                auto& task_queue = m_worker_tasks[i];
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
        std::vector<LockedQueue<WorkerFn>>                                                      m_worker_tasks;
        std::vector<std::mutex>                                                                 m_worker_mutexes;
        alignas(detail::hardware_destructive_interference_size) std::atomic<bool>               m_stop;
        alignas(detail::hardware_destructive_interference_size) std::atomic<std::size_t>        m_pending_tasks;
        std::vector<std::condition_variable>                                                    m_cvs;
        std::size_t                                                                             m_current_worker{0};
        std::condition_variable                                                                 m_wait_cv;
        std::mutex                                                                              m_wait_mtx;
    };

} // namespace fastllama


#endif // FAST_LLAMA_POOL_HPP
