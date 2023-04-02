#if !defined(FAST_LLAMA_FILE_READER_HPP)
#define FAST_LLAMA_FILE_READER_HPP

#include <string>
#include <cstdio>
#include <optional>
#include <type_traits>

namespace fastllama{
    
    struct BinaryFileReader {
        enum class SeekReference {
            Begin = SEEK_SET,
            Current = SEEK_CUR,
            End = SEEK_END
        };

        constexpr BinaryFileReader(std::string_view path) noexcept
            : m_file_ptr(fopen(path.data(), "rb"))
        {}
        
        constexpr BinaryFileReader(BinaryFileReader const& other) noexcept
            : m_file_ptr(other.m_file_ptr)
        {}

        constexpr BinaryFileReader(BinaryFileReader&& other) noexcept 
            : m_file_ptr(other.m_file_ptr)
        {
            if (this != &other) {
                other.m_file_ptr = nullptr;
            }
        }
        
        constexpr BinaryFileReader& operator=(BinaryFileReader&& other) noexcept {
            this->close();
            swap(other, *this);
            return *this;
        }
        
        BinaryFileReader& operator=(BinaryFileReader const& other) noexcept {
            this->close();
            auto temp = BinaryFileReader(other);
            swap(temp, *this);
            return *this;
        }

        template<typename T>
        auto read(T* val, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = sizeof(std::decay_t<T>);
            auto read_count = fread(reinterpret_cast<void *>(val), size, num_of_objects, m_file_ptr);
            return read_count == num_of_objects;
        }
        
        auto read(void* val, std::size_t type_size_in_bytes, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = type_size_in_bytes;
            auto read_count = fread(reinterpret_cast<void *>(val), size, num_of_objects, m_file_ptr);
            return read_count == num_of_objects;
        }

        constexpr operator bool() const noexcept {
            return static_cast<bool>(m_file_ptr);
        }

        constexpr auto tell() const noexcept -> std::optional<std::size_t> {
            auto tell_val = ftell(m_file_ptr);
            if (tell_val < 0) return std::nullopt;
            return static_cast<std::size_t>(tell_val);
        }

        constexpr auto seek(std::size_t offset, SeekReference ref = SeekReference::Begin) noexcept -> bool {
            return fseek(m_file_ptr, offset, static_cast<int>(ref)) == 0;
        }

        constexpr auto eof() const noexcept -> bool {
            return feof(m_file_ptr);
        }

        constexpr auto close() noexcept -> void {
            if (m_file_ptr) fclose(m_file_ptr);
            m_file_ptr = nullptr;
        }

        friend void swap(BinaryFileReader& lhs, BinaryFileReader& rhs) noexcept {
            std::swap(lhs.m_file_ptr, rhs.m_file_ptr);
        }

        ~BinaryFileReader() noexcept {
            if (m_file_ptr) fclose(m_file_ptr);
        }
    private:
        FILE* m_file_ptr;
    };

} // namespace fastllama


#endif // FAST_LLAMA_FILE_READER_HPP
