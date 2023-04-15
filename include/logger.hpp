#if !defined(FAST_LLAMA_LOGGER_HPP)
#define FAST_LLAMA_LOGGER_HPP

#include <string>
#include <sstream>
#include <iostream>
#include <functional>

namespace fastllama {

    struct DefaultLogger {
        using LoggerFunction = std::function<void(char const*, int, char const*, int)>;
        using LoggerResetFunction = std::function<void()>;

        DefaultLogger() noexcept = default;
        DefaultLogger(DefaultLogger const&) = delete;
        DefaultLogger(DefaultLogger && other) noexcept = default;
        DefaultLogger& operator=(DefaultLogger const& other) noexcept = delete;
        DefaultLogger& operator=(DefaultLogger && other) noexcept = default;
        ~DefaultLogger() noexcept = default;

        static void log_func(char const* func_name, int func_name_size, char const* message, int message_size) {
            printf("\x1b[32;1m[Info]:\x1b[0m \x1b[32mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
            fflush(stdout);
        }

        static void log_err_func(char const* func_name, int func_name_size, char const* message, int message_size) {
            fprintf(stderr, "\x1b[31;1m[Error]:\x1b[0m \x1b[31mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
            fflush(stdout);
        }
        
        static void log_warn_func(char const* func_name, int func_name_size, char const* message, int message_size) {
            printf("\x1b[93;1m[Warn]:\x1b[0m \x1b[93mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
            fflush(stdout);
        }
        static void log_reset_func() {
            printf("\x1b[0m");
            fflush(stdout);
        }

        LoggerFunction log{&DefaultLogger::log_func};
        LoggerFunction log_err{&DefaultLogger::log_err_func};
        LoggerFunction log_warn{&DefaultLogger::log_warn_func};
        LoggerResetFunction reset{&DefaultLogger::log_reset_func};
    };
    
    struct Logger {
        using LoggerFunction = typename DefaultLogger::LoggerFunction;

        Logger() noexcept = default;
        Logger(Logger const&) noexcept = delete;
        Logger(Logger &&) noexcept = default;
        Logger& operator=(Logger const&) noexcept = delete;
        Logger& operator=(Logger &&) noexcept = default;
        ~Logger() noexcept = default;

        Logger(DefaultLogger sink) noexcept
            : m_sink(std::move(sink))
        {}

        void reset() const {
            if (!m_sink.reset) return;
            m_sink.reset();
        }

        template<typename... Args>
        void log(std::string_view func_name, Args&&... args) const {
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink.log(func_name.data(), static_cast<int>(func_name.size()), message.data(), static_cast<int>(message.size()));
        }
        
        template<typename... Args>
        void log_err(std::string_view func_name, Args&&... args) const {
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink.log_err(func_name.data(), static_cast<int>(func_name.size()), message.data(), static_cast<int>(message.size()));
        }
        
        template<typename... Args>
        void log_warn(std::string_view func_name, Args&&... args) const {
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink.log_warn(func_name.data(), static_cast<int>(func_name.size()), message.data(), static_cast<int>(message.size()));
        }
    private:
        DefaultLogger m_sink{};
    };

} // namespace fastllama


#endif // FAST_LLAMA_LOGGER_HPP
