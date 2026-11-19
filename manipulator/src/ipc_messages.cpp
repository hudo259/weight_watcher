#include "ipc_messages.hpp"

namespace ipc {

    // Point Cloud 

    std::vector<uint8_t> serialize_point_cloud(
        uint32_t cloud_id,
        const std::vector<glm::vec3>& points,
        const std::vector<glm::vec3>* colors,
        const std::vector<glm::vec3>* normals)
    {
        uint8_t flags = 0;
        if (colors  && !colors->empty())  flags |= 0x01;
        if (normals && !normals->empty()) flags |= 0x02;

        BinaryWriter w;
        w.write_u8 (static_cast<uint8_t>(MsgType::POINT_CLOUD));
        w.write_u32(cloud_id);
        w.write_u8 (flags);
        w.write_u32(static_cast<uint32_t>(points.size()));
        w.write_array(points);
        if (flags & 0x01) w.write_array(*colors);
        if (flags & 0x02) w.write_array(*normals);
        return w.take();
    }

    PointCloudMsg deserialize_point_cloud(BinaryReader& r) {
        PointCloudMsg msg;
        msg.cloud_id = r.read_u32();
        uint8_t flags = r.read_u8();
        uint32_t n    = r.read_u32();
        msg.points = r.read_vec3_array(n);
        if (flags & 0x01) msg.colors  = r.read_vec3_array(n);
        if (flags & 0x02) msg.normals = r.read_vec3_array(n);
        return msg;
    }

    // Layer Meta 

    LayerMetaMsg deserialize_layer_meta(BinaryReader& r) {
        LayerMetaMsg msg;
        msg.cloud_id = r.read_u32();

        uint32_t num_layers = r.read_u32();

        msg.num_classes = r.read_u32();

        msg.point_counts.resize(num_layers);
        r.read_bytes(msg.point_counts.data(), num_layers * sizeof(uint32_t));

        msg.colors.resize(num_layers);
        r.read_bytes(msg.colors.data(), num_layers * sizeof(glm::vec3));

        uint32_t names_total_len = r.read_u32();
        (void)names_total_len;

        msg.names.reserve(num_layers);
        for (uint32_t i = 0; i < num_layers; ++i)
            msg.names.push_back(r.read_string_null_terminated());

        uint32_t types_total_len = r.read_u32();
        (void)types_total_len;

        //worst case: each layer is a different type
        msg.layer_types.reserve(num_layers);
        for(uint32_t i = 0; i < num_layers; ++i) 
            msg.layer_types.push_back(r.read_string_null_terminated());

        return msg;
    }

    // Layout
    LayoutMsg deserialize_layout(BinaryReader& r) {
        LayoutMsg msg;
        msg.layout_id = r.read_u32();

        uint32_t num_layers = r.read_u32();

        msg.name = r.read_string_null_terminated();

        msg.offsets = r.read_vec3_array(num_layers);

        msg.axis_x = r.read_string_null_terminated();
        msg.axis_y = r.read_string_null_terminated();
        msg.axis_z = r.read_string_null_terminated();
        return msg;
    }

    // Transformed Points

    std::vector<uint8_t> serialize_transformed_points(
        uint32_t cloud_id,
        const std::vector<uint32_t>&  indices,
        const std::vector<glm::vec3>& positions)
    {
        BinaryWriter w;
        w.write_u8 (static_cast<uint8_t>(MsgType::TRANSFORMED_POINTS));
        w.write_u32(cloud_id);
        w.write_u32(static_cast<uint32_t>(indices.size()));
        w.write_array(indices);
        w.write_array(positions);
        return w.take();
    }

    TransformedPointsMsg deserialize_transformed_points(BinaryReader& r) {
        TransformedPointsMsg msg;
        msg.cloud_id = r.read_u32();
        uint32_t count = r.read_u32();
        msg.indices   = r.read_u32_array(count);
        msg.positions = r.read_vec3_array(count);
        return msg;
    }

    std::vector<uint8_t> serialize_transformed_points_batch(const TransformedPointsBatchMsg& msg) {
        BinaryWriter w;
        w.write_u8(static_cast<uint8_t>(MsgType::TRANSFORMED_POINTS));
        w.write_u32(static_cast<uint32_t>(msg.clouds.size()));
        for (const auto& cloud : msg.clouds) {
            assert(cloud.indices.size() <= UINT32_MAX);
            w.write_u32(cloud.cloud_id);
            w.write_u32(static_cast<uint32_t>(cloud.indices.size()));
            w.write_array(cloud.indices);
            w.write_array(cloud.positions);
        }
        return w.take();
    }

    TransformedPointsBatchMsg deserialize_transformed_points_batch(BinaryReader& r) {
        TransformedPointsBatchMsg batch;
        uint32_t num_clouds = r.read_u32();
        batch.clouds.reserve(num_clouds);
        for (uint32_t i = 0; i < num_clouds; i++) {
            TransformedPointsMsg entry;
            entry.cloud_id = r.read_u32();
            uint32_t count = r.read_u32();
            entry.indices   = r.read_u32_array(count);
            entry.positions = r.read_vec3_array(count);
            batch.clouds.push_back(std::move(entry));
        }
        return batch;
    }

    // Vector List

    std::vector<uint8_t> serialize_vector_list(
        uint32_t id, uint32_t dimensions, const std::vector<float>& data)
    {
        uint32_t count = dimensions > 0
                         ? static_cast<uint32_t>(data.size() / dimensions) : 0;
        BinaryWriter w;
        w.write_u8 (static_cast<uint8_t>(MsgType::VECTOR_LIST));
        w.write_u32(id);
        w.write_u32(dimensions);
        w.write_u32(count);
        w.write_array(data);
        return w.take();
    }

    VectorListMsg deserialize_vector_list(BinaryReader& r) {
        VectorListMsg msg;
        msg.id         = r.read_u32();
        msg.dimensions = r.read_u32();
        uint32_t count = r.read_u32();
        msg.data       = r.read_f32_array(count * msg.dimensions);
        return msg;
    }

    // Metrics 

    std::vector<uint8_t> serialize_metrics(
        const std::vector<std::string>& names, const std::vector<float>& values)
    {
        uint32_t str_total_len = 0;
        for (const auto& n : names) str_total_len += (uint32_t)(n.size() + 1);

        BinaryWriter w;
        w.write_u8 (static_cast<uint8_t>(MsgType::METRICS));
        w.write_u32(static_cast<uint32_t>(names.size()));
        w.write_u32(str_total_len);
        for (const auto& n : names) w.write_string_null_terminated(n);
        w.write_array(values);
        return w.take();
    }

    MetricsMsg deserialize_metrics(BinaryReader& r) {
        MetricsMsg msg;
        uint32_t count         = r.read_u32();
        uint32_t str_total_len = r.read_u32();
        (void)str_total_len;
        msg.names.reserve(count);
        for (uint32_t i = 0; i < count; i++)
            msg.names.push_back(r.read_string_null_terminated());
        msg.values = r.read_f32_array(count);
        return msg;
    }

    // Images

    ImageMsg deserialize_image_data(BinaryReader& r) {
        ImageMsg msg;
        msg.id       = r.read_u32();
        msg.width    = r.read_u32();
        msg.height   = r.read_u32();
        msg.channels = r.read_u8();
        msg.format   = static_cast<ImageFormat>(r.read_u8());

        // Arbitrary-length null-terminated name
        msg.name = r.read_string_null_terminated();

        size_t data_size = msg.width * msg.height * msg.channels;
        msg.data.resize(data_size);

        r.read_bytes(msg.data.data(), data_size);
        return msg;
    }

    ImageMsg deserialize_image(BinaryReader& r) {
        return deserialize_image_data(r);
    }

    ImageBatchMsg deserialize_image_batch(BinaryReader& r) {
        ImageBatchMsg msg;
        // Wire format: batch_type is stored as uint32_t (was batch_id, same width).
        msg.batch_type = static_cast<ImageBatchType>(r.read_u32());
        uint32_t count = r.read_u32();
        msg.images.reserve(count);
        for (uint32_t i = 0; i < count; i++)
            msg.images.push_back(deserialize_image_data(r));
        return msg;
    }

    // ACK 

    std::vector<uint8_t> serialize_ack() {
        BinaryWriter w(1);
        w.write_u8(static_cast<uint8_t>(MsgType::ACK));
        return w.take();
    }
}