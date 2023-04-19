#if !defined(FAST_LLAMA_MACRO_HPP)
#define FAST_LLAMA_MACRO_HPP

#if defined(_MSC_VER)
    #define FASTLLAMA_ALWAYS_INLINE __forceinline
#else
    #define FASTLLAMA_ALWAYS_INLINE __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
    #define FASTLLAMA_FORMAT_STRING
#else
    #define FASTLLAMA_FORMAT_STRING(fmt_pos, args_pos) __attribute__ ((format (printf, (fmt_pos), (args_pos))))
#endif

#endif // FAST_LLAMA_MACRO_HPP
