#if !defined(FAST_LLAMA_LOGGER_HPP)
#define FAST_LLAMA_LOGGER_HPP

#include <string>
#include <sstream>
#include <iostream>
#include <functional>
#include "macro.hpp"

namespace fastllama {

    struct DefaultLogger {
        using LoggerFunction = std::function<void(char const*, int, char const*, int)>;
        using LoggerResetFunction = std::function<void()>;
        using ProgressCallback = std::function<void(std::size_t, std::size_t)>;

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

        static void progress_func(std::size_t done, std::size_t total) {
            auto perc = (static_cast<float>(done) / static_cast<float>(total)) * 100.0f;
            auto perc_int = static_cast<int>(perc);

            if (perc_int % 8 == 0) {
                printf(".");
                fflush(stdout);
            }

            if (perc_int == 100) {
                printf("\n");
                fflush(stdout);
            }
        }

        LoggerFunction log{&DefaultLogger::log_func};
        LoggerFunction log_err{&DefaultLogger::log_err_func};
        LoggerFunction log_warn{&DefaultLogger::log_warn_func};
        LoggerResetFunction reset{&DefaultLogger::log_reset_func};
        ProgressCallback progress{&DefaultLogger::progress_func};
    };

    struct NullLogger : DefaultLogger {

        NullLogger() noexcept {
            DefaultLogger::log = [](char const*, int, char const*, int) {};
            DefaultLogger::log_err = [](char const*, int, char const*, int) {};
            DefaultLogger::log_warn = [](char const*, int, char const*, int) {};
            DefaultLogger::reset = []() {};
            DefaultLogger::progress = [](std::size_t, std::size_t) {};
        }
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

        static Logger& get_null_logger() noexcept {
            static Logger logger = Logger{ NullLogger{} };
            return logger;
        }

        static Logger& get_default_logger() noexcept {
            static Logger logger = Logger{ DefaultLogger{} };
            return logger;
        }

        void reset() const {
            if (!m_sink.reset) return;
            m_sink.reset();
        }

        template<typename... Args>
        void log(std::string_view func_name, Args&&... args) const {
            if (!m_sink.log) return;
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink.log(func_name.data(), static_cast<int>(func_name.size()), message.data(), static_cast<int>(message.size()));
        }
        
        template<typename... Args>
        void log_err(std::string_view func_name, Args&&... args) const {
            if (!m_sink.log_err) return;
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink.log_err(func_name.data(), static_cast<int>(func_name.size()), message.data(), static_cast<int>(message.size()));
        }
        
        template<typename... Args>
        void log_warn(std::string_view func_name, Args&&... args) const {
            if (!m_sink.log_warn) return;
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink.log_warn(func_name.data(), static_cast<int>(func_name.size()), message.data(), static_cast<int>(message.size()));
        }

        void progress(std::size_t done, std::size_t total) const {
            if (!m_sink.progress) return;
            m_sink.progress(done, total);
        }
    private:
        DefaultLogger m_sink{};
    };

    template<std::size_t N, typename... Ts>
    std::string_view format_str(char (&buff)[N], char const* fmt, Ts&&... args) {
        auto size = snprintf(buff, N, fmt, std::forward<Ts>(args)...);
        return { buff, static_cast<std::size_t>(size) };
    }

    template<typename... Ts>
    std::string_view format_str(char *buff, std::size_t n, char const* fmt, Ts&&... args) {
        auto size = snprintf(buff, n, fmt, std::forward<Ts>(args)...);
        return { buff, static_cast<std::size_t>(size) };
    }

    template<typename... Ts>
    std::string dyn_format_str(char const* fmt, Ts&&... args) {
        std::string buff(256, '\0');
        auto size = sprintf(buff.data(), fmt, std::forward<Ts>(args)...);
        buff.resize(static_cast<std::size_t>(size));
        return std::move(buff);
    }

} // namespace fastllama


#endif // FAST_LLAMA_LOGGER_HPP
