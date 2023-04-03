#if !defined(FAST_LLAMA_LOGGER_HPP)
#define FAST_LLAMA_LOGGER_HPP

#include <string>
#include <sstream>
#include <iostream>

namespace fastllama {

    struct DefaultLogger {
        using LoggerFunction = void(*)(char const*, int, char const*, int);

        static void log_func(char const* func_name, int func_name_size, char const* message, int message_size) {
            printf("\x1b[32;1m[Info]:\x1b[0m \x1b[32mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
        }
        
        static void log_err_func(char const* func_name, int func_name_size, char const* message, int message_size) {
            fprintf(stderr, "\x1b[31;1m[Error]:\x1b[0m \x1b[31mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
        }
        
        static void log_warn_func(char const* func_name, int func_name_size, char const* message, int message_size) {
            printf("\x1b[93;1m[Warn]:\x1b[0m \x1b[93mFunc('%.*s') %.*s\x1b[0m", func_name_size, func_name, message_size, message);
        }

        LoggerFunction log{ &DefaultLogger::log_func };
        LoggerFunction log_err{ &DefaultLogger::log_err_func };
        LoggerFunction log_warn{ &DefaultLogger::log_warn_func };
    };
    
    struct Logger {
        using LoggerFunction = typename DefaultLogger::LoggerFunction;

        Logger(void const* sink = nullptr) noexcept
            : m_sink(&Logger::s_fallback_sink)
        {
            if (sink != nullptr) m_sink = reinterpret_cast<DefaultLogger const*>(sink);
        }

        template<typename... Args>
        void log(std::string_view func_name, Args&&... args) const {
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink->log(func_name.data(), func_name.size(), message.data(), message.size());
        }
        
        template<typename... Args>
        void log_err(std::string_view func_name, Args&&... args) const {
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink->log_err(func_name.data(), func_name.size(), message.data(), message.size());
        }
        
        template<typename... Args>
        void log_warn(std::string_view func_name, Args&&... args) const {
            std::stringstream ss;
            ((ss << args), ...);
            auto message = ss.str();
            m_sink->log_warn(func_name.data(), func_name.size(), message.data(), message.size());
        }
    private:
        inline static DefaultLogger s_fallback_sink = DefaultLogger{};
        DefaultLogger const* m_sink{nullptr};
    };

} // namespace fastllama


#endif // FAST_LLAMA_LOGGER_HPP
