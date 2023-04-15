#if !defined(FAST_LLAMA_FILE_WRITER_HPP)
#define FAST_LLAMA_FILE_WRITER_HPP

#include <string>
#include <cstdio>
#include <optional>
#include <type_traits>

namespace fastllama{
    
    struct BinaryFileWriter {
        enum class SeekReference {
            Begin = SEEK_SET,
            Current = SEEK_CUR,
            End = SEEK_END
        };

        BinaryFileWriter(std::string_view path) noexcept
            : m_file_ptr(fopen(path.data(), "wb"))
        {}
        
        BinaryFileWriter(BinaryFileWriter const& other) noexcept
            : m_file_ptr(other.m_file_ptr)
        {}

        BinaryFileWriter(BinaryFileWriter&& other) noexcept 
            : m_file_ptr(other.m_file_ptr)
        {
            if (this != &other) {
                other.m_file_ptr = nullptr;
            }
        }
        
        BinaryFileWriter& operator=(BinaryFileWriter&& other) noexcept {
            this->close();
            swap(other, *this);
            return *this;
        }
        
        BinaryFileWriter& operator=(BinaryFileWriter const& other) noexcept {
            this->close();
            auto temp = BinaryFileWriter(other);
            swap(temp, *this);
            return *this;
        }

        template<typename T>
        auto write(T const* val, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = sizeof(std::decay_t<T>);
            auto read_count = fwrite(reinterpret_cast<void const*>(val), size, num_of_objects, m_file_ptr);
            return read_count == num_of_objects;
        }
        
        auto write(void const* val, std::size_t type_size_in_bytes, std::size_t num_of_objects = 1) noexcept -> bool {
            auto const size = type_size_in_bytes;
            auto read_count = fwrite(reinterpret_cast<void const*>(val), size, num_of_objects, m_file_ptr);
            return read_count == num_of_objects;
        }

        constexpr operator bool() const noexcept {
            return static_cast<bool>(m_file_ptr);
        }

        auto tell() const noexcept -> std::optional<std::size_t> {
            auto tell_val = ftell(m_file_ptr);
            if (tell_val < 0) return std::nullopt;
            return static_cast<std::size_t>(tell_val);
        }

        auto seek(std::size_t offset, SeekReference ref = SeekReference::Begin) noexcept -> bool {
            return fseek(m_file_ptr, static_cast<long>(offset), static_cast<int>(ref)) == 0;
        }

        auto eof() const noexcept -> bool {
            return feof(m_file_ptr);
        }

        auto close() noexcept -> void {
            if (m_file_ptr) fclose(m_file_ptr);
            m_file_ptr = nullptr;
        }

        auto set_buffer(char* buffer, std::size_t size) noexcept {
            setvbuf(m_file_ptr, buffer, _IOFBF, size);
        }

        friend void swap(BinaryFileWriter& lhs, BinaryFileWriter& rhs) noexcept {
            std::swap(lhs.m_file_ptr, rhs.m_file_ptr);
        }

        ~BinaryFileWriter() noexcept {
            if (m_file_ptr) fclose(m_file_ptr);
        }
    private:
        FILE* m_file_ptr;
    };

} // namespace fastllama


#endif // FAST_LLAMA_FILE_WRITER_HPP
