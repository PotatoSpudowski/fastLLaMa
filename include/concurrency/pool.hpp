#if !defined(FAST_LLAMA_POOL_HPP)
#define FAST_LLAMA_POOL_HPP

#include <thread>
#include "deque.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace fastllama {

    template <typename W>
    struct ThreadPool {
        static_assert(std::is_invocable_v<W>, "Type 'W' should invocable");

        struct Worker;

        using size_type = std::size_t;

        ThreadPool(size_type workers = std::thread::hardware_concurrency())
            : m_mu(workers)
            , m_cv(workers)
        {
            for(auto i = 0ul; i < workers; ++i) {
                m_workers.emplace_back(this, i);
                m_threads.emplace_back([&worker=m_workers[i], &mu=m_mu[i], &cv=m_cv[i], &stop=m_stop, this] {
                    worker.run(stop, mu, cv, [this](size_type id){ return this->try_steal(id); });
                });
            }
        }

        constexpr void stop() noexcept { m_stop = true; }

        void add_work(W&& work) {
            auto idx = m_current_work_receiver_id;
            m_current_work_receiver_id = (m_current_work_receiver_id + 1) % m_workers.size();
            m_workers[idx].add_work(std::move(work));
            m_cv[idx].notify_all();
        }

        void wait() {
            for(auto& t : m_threads) t.join();
        }

        ~ThreadPool() {
            stop();
            for(auto i = 0ul; i < m_workers.size(); ++i) {
                m_cv[i].notify_all();
            }
            wait();
        }

    private:
        auto try_steal(size_type id) noexcept -> std::optional<W> {
            for(auto i = std::size_t{}; i < m_workers.size(); ++i) {
                if (i == id) continue;
                auto& worker = m_workers[i];
                auto item = worker.try_steal_local_work();
                if (item) return item;
            }
            return {};
        }
    private:
        std::vector<std::mutex> m_mu;
        std::vector<std::condition_variable> m_cv;
        std::vector<Worker> m_workers;
        std::vector<std::thread> m_threads;
        size_type m_current_work_receiver_id{}; // Used for round robin work so that all threads get some work
        bool m_stop{false};
    };

    template <typename W>
    struct ThreadPool<W>::Worker {
        using size_type = std::size_t;

        Worker() = default;
        
        Worker(ThreadPool* pool, size_type worker_id)
            : m_worker_id(worker_id)
        {}

        Worker(Worker&& other) noexcept
            : m_local_queue(std::move(other.m_local_queue))
            , m_pool(std::move(other.m_pool))
            , m_worker_id(std::move(other.m_worker_id))
        {}
        Worker& operator=(Worker&& other) noexcept {
            auto temp = Worker(std::move(other));
            swap(*this, temp);
            return *this;
        }

        ~Worker() = default;

        template<typename Fn>
        void run(bool const& stop, std::mutex& mu, std::condition_variable& cv, Fn&& try_steal_from_worker_fn) noexcept {
            auto i = 0;
            while (++i < 20) {
                {
                    std::unique_lock lk{mu};
                    cv.wait(lk, [this, &stop, &try_steal_from_worker_fn]{
                        if (m_local_queue.empty()) {
                            auto item = try_steal_from_worker_fn(m_worker_id);
                            if (item) this->m_local_queue.emplace(std::move(*item));
                        }
                        return stop || !m_local_queue.empty();
                    });
                }

                if (m_local_queue.empty()) {
                    if (stop) break;
                    auto item = try_steal_from_worker_fn(m_worker_id);
                    if (item) this->m_local_queue.emplace(std::move(*item));
                }

                while(!m_local_queue.empty()) {
                    auto task_maybe = m_local_queue.pop();
                    if (!task_maybe) break;

                    std::invoke(std::move(*task_maybe));
                }
            }
        }

        void add_work(W&& work) noexcept {
            m_local_queue.emplace(std::move(work));
            // std::this_thread::sleep_for(1s);
            // std::cout<<"Id: "<<m_worker_id<<", Size: "<<m_local_queue.size()<<", Empty: "<<m_local_queue.empty()<<std::endl;
            // exit(0);
        }

        std::optional<W> try_steal_local_work() noexcept {
            return std::move(m_local_queue.steal());
        }

        constexpr void set_worker_id(size_type id) noexcept { m_worker_id = id; }

        friend void swap(Worker& lhs, Worker& rhs) noexcept {
            std::swap(lhs.m_local_queue, rhs.m_local_queue);
            std::swap(lhs.m_pool, rhs.m_pool);
            std::swap(lhs.m_worker_id, rhs.m_worker_id);
        }
    private:
        ThreadPool*             m_pool{nullptr};
        Deque<W>                m_local_queue{};
        size_type               m_worker_id{};
    };

} // namespace fastllama


#endif // FAST_LLAMA_POOL_HPP
