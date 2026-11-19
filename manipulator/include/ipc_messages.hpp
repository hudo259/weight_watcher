#pragma once

#include <glm/glm.hpp>

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <string>

namespace ipc {

    enum class MsgType : uint8_t {
        POINT_CLOUD        = 1,
        TRANSFORMED_POINTS = 2,
        TRANSFORMED_POINTS_BATCH = 3,
        VECTOR_LIST        = 4,
        METRICS            = 5,
        IMAGE              = 6,
        IMAGE_BATCH        = 7,
        LAYER_META         = 8,
        LAYOUT             = 9,
        ACK                = 255,
    };

    enum class ImageFormat : uint8_t {
        GRAYSCALE = 1,
        RGB       = 2,
        BGR       = 3,
        RGBA      = 4,
        BGRA      = 5
    };

    // Describes the semantic content of an ImageBatchMsg.
    // Stored in the wire-format batch_type field (uint32_t).
    enum class ImageBatchType : uint32_t {
        PREDICTIONS    = 0,  // backend → frontend: prediction overlays, sent after each eval
        ACTIVATION_MAP = 1,  // backend → frontend: per-filter grayscale heatmaps, on demand
        RAW_IMAGES     = 2,  // backend → frontend: raw anchor images, sent once at connect
    };

    constexpr std::string_view get_ImageBatchType_name(ImageBatchType imb) {
        switch (imb)
        {
            case ImageBatchType::PREDICTIONS:
                return "Predictions";
            case ImageBatchType::ACTIVATION_MAP:
                return "Activation Map";
            case ImageBatchType::RAW_IMAGES:
                return "Raw Images";
            default:
                return "Invalid Type";
        }
    }

    namespace VectorListID {
        constexpr uint32_t FEATURES = 1;
        constexpr uint32_t LOSS = 2;
        constexpr uint32_t GRADIENTS = 3;
        constexpr uint32_t CONCEPT_SELECT = 4;  // frontend -> backend: float-cast class index
        constexpr uint32_t CONCEPT_SCORES = 5;  // backend -> frontend: one float per point
        // frontend -> backend: one row per selected point, dimensions=3
        // row format: [cloud_idx, point_idx, aggregation_mode]
        //   aggregation_mode: 0 = average across filters/layers, 1 = maximum
        constexpr uint32_t ACTIVATION_REQ = 6;
        constexpr uint32_t JOB_STATUS = 255;
    }

    // Backend job kinds, sent in JOB_STATUS (must match InferenceEngine._JobType in Python)
    enum class JobType : uint32_t {
        POINT_CLOUD         = 0,
        TRANSFORMED_POINTS  = 1,
        CONCEPT_SELECT      = 2,
        ACTIVATION_REQUEST  = 3,
        RAW_IMAGES          = 4,
        CKA_LAYOUT          = 5,
        TRAJECTORY_GEOMETRY = 6,
        SENSITIVITY_LAYOUT  = 7,
        EV_LAYOUT           = 8,
        CORRELATION_LAYOUT  = 9,
        COUNT
    };

    // Job lifecycle (must match JobStatus in ipc/iipc.py).
    // Every QUEUED is followed by exactly one DONE / FAILED / SKIPPED / DROPPED.
    enum class JobStatus : uint32_t {
        QUEUED  = 1,
        DONE    = 2,
        FAILED  = 3,
        SKIPPED = 4,
        DROPPED = 5,
        RESET   = 6,   // backend cleared its queue on (re)connect
    };

    namespace LayoutID {
        constexpr uint32_t DEPTH = 0;
        constexpr uint32_t CKA = 1;
        constexpr uint32_t TRAJECTORY_GEOMETRY = 2;
        constexpr uint32_t SENSITIVITY = 3;
        constexpr uint32_t EXPLAINED_VARIANCE = 4;
        constexpr uint32_t CORRELATION = 5;
    }

    struct PointCloudMsg {
        uint32_t cloud_id = 0;
        std::vector<glm::vec3> points;
        std::vector<glm::vec3> colors;
        std::vector<glm::vec3> normals;
    };

    struct LayerMetaMsg {
        uint32_t cloud_id;
        uint32_t num_classes;
        std::vector<uint32_t>    point_counts;
        std::vector<glm::vec3>   colors;
        std::vector<std::string> names;
        std::vector<std::string> layer_types;
    };

    struct LayoutMsg {
        uint32_t layout_id;
        std::string name;
        std::vector<glm::vec3> offsets; //one per layer, matches LayerMetaMsg order
        std::string axis_x;
        std::string axis_y;
        std::string axis_z;
    };

    struct TransformedPointsMsg {
        uint32_t cloud_id = 0;
        std::vector<uint32_t>  indices;
        std::vector<glm::vec3> positions;
    };

    struct TransformedPointsBatchMsg {
        //single-cloud transforms are simply a batch of size 1
        std::vector<TransformedPointsMsg> clouds;
    };

    struct VectorListMsg {
        uint32_t id         = 0;
        uint32_t dimensions = 0;
        std::vector<float> data;

        size_t count() const {
            return dimensions > 0 ? data.size() / dimensions : 0;
        }
        float at(size_t i, size_t j) const { return data[i * dimensions + j]; }
        const float* vector(size_t i) const { return &data[i * dimensions]; }
        float value() const { return data.empty() ? 0.0f : data[0]; }
    };

    struct MetricsMsg {
        std::vector<std::string> names;
        std::vector<float>       values;

        size_t count() const { return names.size(); }
        const float* get(const std::string& name) const {
            for (size_t i = 0; i < names.size(); i++)
                if (names[i] == name) return &values[i];
            return nullptr;
        }
    };

    struct ImageMsg {
        uint32_t    id       = 0;
        uint32_t    width    = 0;
        uint32_t    height   = 0;
        uint8_t     channels = 0;
        ImageFormat format   = ImageFormat::BGR;
        std::string name;
        std::vector<uint8_t> data;
    };

    struct ImageBatchMsg {
        ImageBatchType        batch_type = ImageBatchType::PREDICTIONS;
        std::vector<ImageMsg> images;
    };

    // Binary IO stuff
    class BinaryWriter {
    public:
        explicit BinaryWriter(size_t reserve = 256) { buffer_.reserve(reserve); }

        void write_u8 (uint8_t  v) { buffer_.push_back(v); }
        void write_u32(uint32_t v) { append(&v, 4); }
        void write_f32(float    v) { append(&v, 4); }
        void write_vec3(const glm::vec3& v) { write_f32(v.x); write_f32(v.y); write_f32(v.z); }
        void write_bytes(const void* data, size_t size) { append(data, size); }
        void write_string_null_terminated(const std::string& s) { append(s.c_str(), s.size() + 1); }
        template<typename T>
        void write_array(const std::vector<T>& arr) { append(arr.data(), arr.size() * sizeof(T)); }

        const std::vector<uint8_t>& buffer() const { return buffer_; }
        std::vector<uint8_t>&&      take()         { return std::move(buffer_); }
        size_t size() const { return buffer_.size(); }

    private:
        void append(const void* src, size_t n) {
            size_t pos = buffer_.size();
            buffer_.resize(pos + n);
            std::memcpy(&buffer_[pos], src, n);
        }
        std::vector<uint8_t> buffer_;
    };

    class BinaryReader {
    public:
        BinaryReader(const uint8_t* data, size_t size)
            : data_(data), size_(size), pos_(0) {}

        bool   has_remaining(size_t n) const { return pos_ + n <= size_; }
        size_t remaining()             const { return size_ - pos_; }

        uint8_t  read_u8()  { check(1); return data_[pos_++]; }
        uint32_t read_u32() { check(4); uint32_t v; std::memcpy(&v, &data_[pos_], 4); pos_ += 4; return v; }
        float    read_f32() { check(4); float    v; std::memcpy(&v, &data_[pos_], 4); pos_ += 4; return v; }
        glm::vec3 read_vec3() { return { read_f32(), read_f32(), read_f32() }; }

        void read_bytes(void* dest, size_t n) { check(n); std::memcpy(dest, &data_[pos_], n); pos_ += n; }

        std::string read_string_null_terminated() {
            const char* start = reinterpret_cast<const char*>(&data_[pos_]);
            size_t len = 0;
            while (pos_ + len < size_ && data_[pos_ + len] != '\0') len++;
            check(len + 1);
            pos_ += len + 1;
            return std::string(start, len);
        }

        std::vector<uint32_t> read_u32_array(size_t count) {
            std::vector<uint32_t> arr(count);
            read_bytes(arr.data(), count * sizeof(uint32_t));
            return arr;
        }
        std::vector<float> read_f32_array(size_t count) {
            std::vector<float> arr(count);
            read_bytes(arr.data(), count * sizeof(float));
            return arr;
        }
        std::vector<glm::vec3> read_vec3_array(size_t count) {
            std::vector<glm::vec3> arr(count);
            read_bytes(arr.data(), count * sizeof(glm::vec3));
            return arr;
        }

    private:
        void check(size_t n) {
            if (!has_remaining(n)) throw std::runtime_error("BinaryReader: unexpected end of data");
        }
        const uint8_t* data_;
        size_t         size_;
        size_t         pos_;
    };

    // Serialization
    std::vector<uint8_t> serialize_point_cloud(
        uint32_t cloud_id,
        const std::vector<glm::vec3>& points,
        const std::vector<glm::vec3>* colors  = nullptr,
        const std::vector<glm::vec3>* normals = nullptr);
    PointCloudMsg deserialize_point_cloud(BinaryReader& r);

    LayerMetaMsg deserialize_layer_meta(BinaryReader& r);

    LayoutMsg deserialize_layout(BinaryReader& r);

    std::vector<uint8_t> serialize_transformed_points(
        uint32_t cloud_id,
        const std::vector<uint32_t>&  indices,
        const std::vector<glm::vec3>& positions);
    std::vector<uint8_t> serialize_transformed_points_batch(const TransformedPointsBatchMsg& msg);

    TransformedPointsMsg deserialize_transformed_points(BinaryReader& r);
    TransformedPointsBatchMsg deserialize_transformed_points_batch(BinaryReader& r);

    std::vector<uint8_t> serialize_vector_list(
        uint32_t id, uint32_t dimensions, const std::vector<float>& data);
    inline std::vector<uint8_t> serialize_loss(float v) {
        return serialize_vector_list(VectorListID::LOSS, 1, {v});
    }
    VectorListMsg deserialize_vector_list(BinaryReader& r);

    std::vector<uint8_t> serialize_metrics(
        const std::vector<std::string>& names, const std::vector<float>& values);
    MetricsMsg deserialize_metrics(BinaryReader& r);

    ImageMsg      deserialize_image_data (BinaryReader& r);
    ImageMsg      deserialize_image      (BinaryReader& r);
    ImageBatchMsg deserialize_image_batch(BinaryReader& r);

    std::vector<uint8_t> serialize_ack();
}
