#if !defined(FAST_LLAMA_MACRO_HPP)
#define FAST_LLAMA_MACRO_HPP

#if defined(_MSC_VER)
    #define FASTLLAMA_ALWAYS_INLINE __forceinline
#else
    #define FASTLLAMA_ALWAYS_INLINE __attribute__((always_inline))
#endif

#endif // FAST_LLAMA_MACRO_HPP
