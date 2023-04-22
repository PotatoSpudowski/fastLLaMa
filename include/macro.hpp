#if !defined(FAST_LLAMA_MACRO_HPP)
#define FAST_LLAMA_MACRO_HPP

#if defined(_MSC_VER)
    #define FAST_LLAMA_ALWAYS_INLINE __forceinline
#else
    #define FAST_LLAMA_ALWAYS_INLINE __attribute__((always_inline))
#endif

#if !defined(__GNUC__)
    #define FAST_LLAMA_FORMAT_STRING
#else
    #ifdef __MINGW32__
        #define FAST_LLAMA_FORMAT_STRING(fmt_pos, args_pos) __attribute__ ((format (gnu_printf, (fmt_pos), (args_pos))))
    #else
        #define FAST_LLAMA_FORMAT_STRING(fmt_pos, args_pos) __attribute__ ((format (printf, (fmt_pos), (args_pos))))
    #endif
#endif

#define _FAST_LLAMA_ASSERT_NO_MESSAGE(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "FAST_LLAMA_ASSERT: %s:%d: Expr(%s).\n", __FILE__, __LINE__, #x); \
            std::abort(); \
        } \
    } while (false)

#define _FAST_LLAMA_ASSERT_MESSAGE(x, message) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "FAST_LLAMA_ASSERT: %s:%d: Expr(%s). \"%s\"\n", __FILE__, __LINE__, #x, message); \
            std::abort(); \
        } \
    } while (false)


#define _FAST_LLAMA_ASSERT_GET_MACRO(_1, _2, NAME, ...) NAME
#define FAST_LLAMA_ASSERT(...) _FAST_LLAMA_ASSERT_GET_MACRO(__VA_ARGS__, _FAST_LLAMA_ASSERT_MESSAGE, _FAST_LLAMA_ASSERT_NO_MESSAGE)(__VA_ARGS__)
    
#endif // FAST_LLAMA_MACRO_HPP
