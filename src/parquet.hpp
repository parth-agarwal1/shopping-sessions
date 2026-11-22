// Minimal flat, uncompressed Parquet writer. Format: parquet.thrift and the
// Apache Parquet PLAIN and RLE encoding specifications.
#pragma once
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace parquet {
using Bytes = std::vector<uint8_t>;
enum Type { Int32 = 1, Int64 = 2, String = 6, Date = 100 };
struct Field { std::string_view name; Type type; bool nullable = false; };

inline void varint(Bytes& bytes, uint64_t n) {
    while (n >= 128) {
        bytes.push_back(uint8_t(n) | 128);
        n >>= 7;
    }
    bytes.push_back(uint8_t(n));
}
inline void little(Bytes& bytes, uint64_t value, unsigned width) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    const auto start=bytes.size();
    bytes.resize(start+width);
    std::memcpy(bytes.data()+start,&value,width);
#else
    for (unsigned i = 0; i < width; ++i) {
        bytes.push_back(uint8_t(value));
        value >>= 8;
    }
#endif
}

namespace thrift {
enum Wire { I32 = 5, I64 = 6, Binary = 8, List = 9, Struct = 12 };
inline void integer(Bytes& bytes, int64_t value) {
    varint(bytes, (uint64_t(value) << 1) ^ uint64_t(-(value < 0)));
}
inline void binary(Bytes& bytes, std::string_view value) {
    varint(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}
struct Record {
    Bytes& bytes;
    int last = 0;
    void header(int id, Wire type) {
        int delta = id - last;
        if (delta > 0 && delta < 16) bytes.push_back(uint8_t(delta * 16 + type));
        else { bytes.push_back(type); integer(bytes, id); }
        last = id;
    }
    void number(int id, int64_t value, Wire type = I32) {
        header(id, type);
        integer(bytes, value);
    }
    void text(int id, std::string_view value) {
        header(id, Binary);
        binary(bytes, value);
    }
    void list(int id, Wire kind, size_t count) {
        header(id, List);
        bytes.push_back(uint8_t((count < 15 ? count : 15) * 16 + kind));
        if (count >= 15) varint(bytes, count);
    }
    void end() { bytes.push_back(0); }
};
} // namespace thrift

class Table {
    struct Column {
        Field schema;
        Bytes values, levels;
        size_t level_count = 0;
        bool present = false;

        void append_level(bool next) {
            if (level_count && next != present) flush_levels();
            present = next;
            ++level_count;
        }
        void flush_levels() {
            if (!level_count) return;
            varint(levels, level_count * 2);
            levels.push_back(present);
            level_count = 0;
        }
    };
    std::vector<Column> columns;
    size_t rows = 0;

    static int physical(Type type) { return type == Date ? Int32 : type; }
    static void send(std::ofstream& f, const Bytes& bytes) {
        if (!bytes.empty()) f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

public:
    explicit Table(std::initializer_list<Field> schema) {
        columns.reserve(schema.size());
        for (auto field : schema) columns.push_back({field, {}, {}});
    }
    size_t size() const { return rows; }
    void integer(size_t column, int64_t value) {
        auto& c = columns[column];
        little(c.values, uint64_t(value), c.schema.type == Int64 ? 8 : 4);
    }
    void string(size_t column, std::string_view value, bool present = true) {
        auto& c = columns[column];
        if (c.schema.nullable) c.append_level(present);
        if (!present) return;
        little(c.values, value.size(), 4);
        c.values.insert(c.values.end(), value.begin(), value.end());
    }
    void finish_row() { ++rows; }

    void flush(const std::filesystem::path& path) {
        using namespace thrift;
        std::ofstream f;
        f.exceptions(std::ios::badbit | std::ios::failbit);
        f.open(path, std::ios::binary);
        f.write("PAR1", 4);
        struct Page { int64_t offset, size; };
        std::vector<Page> pages;
        int64_t offset = 4;
        if (rows) for (auto& c : columns) {
            c.flush_levels();
            const auto data_size = c.values.size() + (c.schema.nullable ? 4 + c.levels.size() : 0);
            Bytes header;
            Record page{header};
            page.number(1, 0); // DATA_PAGE
            page.number(2, data_size);
            page.number(3, data_size);
            page.header(5, Struct);
            Record data{header};
            data.number(1, rows);
            data.number(2, 0); // PLAIN
            data.number(3, 3); // RLE definition levels
            data.number(4, 3); // RLE repetition levels
            data.end();
            page.end();
            send(f, header);
            if (c.schema.nullable) {
                Bytes length;
                little(length, c.levels.size(), 4);
                send(f, length);
                send(f, c.levels);
            }
            send(f, c.values);
            auto size = int64_t(header.size() + data_size);
            pages.push_back({offset, size});
            offset += size;
        }

        Bytes footer;
        Record file{footer};
        file.number(1, 1);
        file.list(2, Struct, columns.size() + 1);
        Record root{footer};
        root.text(4, "schema");
        root.number(5, columns.size());
        root.end();
        for (auto& c : columns) {
            Record field{footer};
            field.number(1, physical(c.schema.type));
            field.number(3, c.schema.nullable ? 1 : 0);
            field.text(4, c.schema.name);
            if (c.schema.type == String) field.number(6, 0); // UTF8
            else if (c.schema.type == Date) field.number(6, 6); // DATE
            field.end();
        }
        file.number(3, rows, I64);
        file.list(4, Struct, rows != 0);
        if (rows) {
            Record group{footer};
            group.list(1, Struct, columns.size());
            for (size_t i = 0; i < columns.size(); ++i) {
                auto field = columns[i].schema;
                auto page = pages[i];
                Record chunk{footer};
                chunk.number(2, 0, I64); // no separate column metadata
                chunk.header(3, Struct);
                Record meta{footer};
                meta.number(1, physical(field.type));
                meta.list(2, I32, 2);
                thrift::integer(footer, 0);
                thrift::integer(footer, 3);
                meta.list(3, Binary, 1);
                binary(footer, field.name);
                meta.number(4, 0); // UNCOMPRESSED
                meta.number(5, rows, I64);
                meta.number(6, page.size, I64);
                meta.number(7, page.size, I64);
                meta.number(9, page.offset, I64);
                meta.end();
                chunk.end();
            }
            group.number(2, offset - 4, I64);
            group.number(3, rows, I64);
            group.end();
        }
        file.end();
        auto footer_size = footer.size();
        little(footer, footer_size, 4);
        footer.insert(footer.end(), {'P','A','R','1'});
        send(f, footer);
        f.close();
        for (auto& c : columns) {
            c.values.clear();
            c.levels.clear();
        }
        rows = 0;
    }
};
} // namespace parquet
