#if !defined(FAST_LLAMA_MMAP_HPP)
#define FAST_LLAMA_MMAP_HPP

#include "macro.hpp"
#include "utils.hpp"
#include "file_reader.hpp"

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <io.h>
#endif

namespace fastllama {
    
    struct MMappedFile {

    #ifdef _POSIX_MAPPED_FILES
        static constexpr bool SUPPORTED = true;

        MMappedFile(BinaryFileReader const* file, bool prefetch = true) noexcept
            : m_size(file->size())
        {
            int fd = fileno(file->handle());
            int flags = MAP_SHARED;
            #if defined(__linux__)
                flags |= MAP_POPULATE;
            #endif

            m_address = mmap(nullptr, m_size, PROT_READ, flags, fd, 0);

            if (m_address == MAP_FAILED) {
                m_address = nullptr;
                return;
            }

            if (prefetch) {
                // Advise the kernel to preload the mapped memory
                if (madvise(m_address, m_size, MADV_WILLNEED) != 0) {
                    fprintf(stderr, "warning: madvise(.., MADV_WILLNEED) failed: %s\n", std::strerror(errno));
                }
            }
        }

        ~MMappedFile() noexcept {
            if (m_address) {
                munmap(m_address, m_size);
            }
        }

    #elif defined(_WIN32)
        static constexpr bool SUPPORTED = true;

        llama_mmap(BinaryFileReader * file, bool prefetch = true)
            : m_size(file->size)
        {

            HANDLE hFile = (HANDLE) _get_osfhandle(_fileno(file->fp));

            HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            DWORD error = GetLastError();

            if (hMapping == NULL) {
                m_address = nullptr;
                return;
            }

            m_address = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
            error = GetLastError();
            CloseHandle(hMapping);

            if (m_address == NULL) {
                m_address = nullptr;
                return;
            }

            #if _WIN32_WINNT >= _WIN32_WINNT_WIN8
            if (prefetch) {
                // Advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = m_address;
                range.NumberOfBytes = (SIZE_T)m_size;
                if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    fprintf(stderr, "warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
            #else
                #pragma message("warning: You are building for pre-Windows 8; prefetch not supported")
            #endif // _WIN32_WINNT >= _WIN32_WINNT_WIN8
        }

        ~llama_mmap() {
            if (!UnmapViewOfFile(m_address)) {
                fprintf(stderr, "warning: UnmapViewOfFile failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
            }
        }

    #else

        static constexpr bool SUPPORTED = false;

        llama_mmap(BinaryFileReader *) noexcept {}

    #endif

        MMappedFile(MMappedFile const& other) noexcept = delete;
        MMappedFile(MMappedFile&& other) noexcept = default;
        MMappedFile& operator=(MMappedFile const& other) noexcept = delete;
        MMappedFile& operator=(MMappedFile&& other) noexcept = default;

        constexpr operator bool() const noexcept { return m_address != nullptr; }

        std::uint8_t* get_data_offset(std::size_t offset) const noexcept {
            FAST_LLAMA_ASSERT(offset <= m_size, "MMappedFile::get_data_offset: offset out of bounds");
            return static_cast<std::uint8_t*>(m_address) + offset;
        }
        
        std::uint8_t* get_data_offset(std::size_t offset) noexcept {
            FAST_LLAMA_ASSERT(offset <= m_size, "MMappedFile::get_data_offset: offset out of bounds");
            return static_cast<std::uint8_t*>(m_address) + offset;
        }

    private:
        void*       m_address{nullptr};
        std::size_t m_size{0};
    };

    // Represents some region of memory being locked using mlock or VirtualLock;
    // will automatically unlock on destruction.

    struct MemoryLock {

        constexpr MemoryLock() noexcept = default;
        MemoryLock(MemoryLock const& other) noexcept = delete;
        MemoryLock(MemoryLock&& other) noexcept = default;
        MemoryLock& operator=(MemoryLock const& other) noexcept = delete;
        MemoryLock& operator=(MemoryLock&& other) noexcept = default;
        
        void init(void* address) noexcept {
            FAST_LLAMA_ASSERT((m_address == nullptr && m_size == 0), "MemoryLock::init called twice");
            m_address = address;
        }

        void grow_to(std::size_t target_size) noexcept {
            FAST_LLAMA_ASSERT(m_address != nullptr, "MemoryLock::grow_to called before init");
            if (m_failed_already) {
                return;
            }
            
            size_t granularity = lock_granularity();
            target_size = (target_size + granularity - 1) & ~(granularity - 1);
            if (target_size > m_size) {
                if (raw_lock((uint8_t *) m_address + m_size, target_size - m_size)) {
                    m_size = target_size;
                } else {
                    m_failed_already = true;
                }
            }
        }

    #ifdef _POSIX_MEMLOCK_RANGE
        static constexpr bool SUPPORTED = true;

        std::size_t lock_granularity() const noexcept {
            return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        }

        #ifdef __APPLE__
            #define MLOCK_SUGGESTION \
                "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
                "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MLOCK (ulimit -l).\n"
        #else
            #define MLOCK_SUGGESTION \
                "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
        #endif

        bool raw_lock(const void * addr, size_t size) noexcept {
            if (!mlock(addr, size)) {
                return true;
            } else {
                char* errmsg = std::strerror(errno);
                bool suggest = (errno == ENOMEM);

                // Check if the resource limit is fine after all
                struct rlimit lock_limit;
                if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit))
                    suggest = false;
                if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size))
                    suggest = false;

                fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                        size, m_size, errmsg, suggest ? MLOCK_SUGGESTION : "");
                return false;
            }
        }

        void raw_unlock(void * addr, size_t size) {
            if (munlock(addr, size)) {
                fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
            }
        }

        #undef MLOCK_SUGGESTION

    #elif defined(_WIN32)

        static constexpr bool SUPPORTED = true;

        std::size_t lock_granularity() const noexcept {
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            return static_cast<std::size_t>(si.dwPageSize);
        }

        bool raw_lock(void * addr, size_t size) {
            for (int tries = 1; ; tries++) {
                if (VirtualLock(addr, size)) {
                    return true;
                }
                if (tries == 2) {
                    fprintf(stderr, "warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                            size, m_size, llama_format_win_err(GetLastError()).c_str());
                    return false;
                }

                // It failed but this was only the first try; increase the working
                // set size and try again.
                SIZE_T min_ws_size, max_ws_size;
                if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                    fprintf(stderr, "warning: GetProcessWorkingSetSize failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                    return false;
                }
                // Per MSDN: "The maximum number of pages that a process can lock
                // is equal to the number of pages in its minimum working set minus
                // a small overhead."
                // Hopefully a megabyte is enough overhead:
                size_t increment = size + 1048576;
                // The minimum must be <= the maximum, so we need to increase both:
                min_ws_size += increment;
                max_ws_size += increment;
                if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                    fprintf(stderr, "warning: SetProcessWorkingSetSize failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                    return false;
                }
            }
        }

        void raw_unlock(void * addr, size_t size) {
            if (!VirtualUnlock(addr, size)) {
                fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
            }
        }

    #else

        static constexpr bool SUPPORTED = false;

        void raw_lock(const void * addr, size_t size) {
            fprintf(stderr, "warning: mlock not supported on this system\n");
        }

        void raw_unlock(const void * addr, size_t size) {}

    #endif
    
    private:
        void*       m_address{nullptr};
        std::size_t m_size{0};
        bool        m_failed_already{false};
    };
    
} // namespace fastllama


#endif // FAST_LLAMA_MMAP_HPP
