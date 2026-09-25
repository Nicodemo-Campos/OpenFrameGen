// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace ofg::vulkan {

enum class ScaleFilter : std::uint32_t {
    Bilinear = 0,
    Bicubic = 1,
};

struct GpuTimingSample {
    double scaler_ms = 0.0;
    double sharpening_ms = 0.0;
    double total_ms = 0.0;
};

struct MotionValidationSample {
    float motion_x = 0.0F;
    float motion_y = 0.0F;
    float mean_error = 0.0F;
    float valid = 0.0F;
};

class VulkanPassthroughPipeline {
public:
    VulkanPassthroughPipeline() = default;
    ~VulkanPassthroughPipeline();

    VulkanPassthroughPipeline(const VulkanPassthroughPipeline&) = delete;
    VulkanPassthroughPipeline& operator=(const VulkanPassthroughPipeline&) =
        delete;

    [[nodiscard]] static bool build_available() noexcept;
    [[nodiscard]] static bool supports_source_format(
        VkFormat format) noexcept;

    [[nodiscard]] bool initialize(
        VkDevice device,
        PFN_vkGetDeviceProcAddr get_device_proc_addr,
        const VkPhysicalDeviceMemoryProperties& memory_properties,
        VkExtent2D source_extent,
        VkExtent2D output_extent,
        VkFormat source_format,
        ScaleFilter filter,
        float sharpening_strength,
        float timestamp_period_ns,
        std::uint32_t timestamp_valid_bits,
        bool motion_validation,
        const std::vector<VkImage>& source_images) noexcept;

    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool sharpening_enabled() const noexcept;
    [[nodiscard]] float sharpening_strength() const noexcept;
    [[nodiscard]] bool timing_enabled() const noexcept;
    [[nodiscard]] bool read_timing(
        std::uint32_t slot_index,
        GpuTimingSample& sample) const noexcept;
    [[nodiscard]] bool frame_history_ready() const noexcept;
    [[nodiscard]] std::uint64_t history_frame_count() const noexcept;
    [[nodiscard]] bool motion_estimation_enabled() const noexcept;
    [[nodiscard]] bool motion_field_ready() const noexcept;
    [[nodiscard]] VkExtent2D motion_field_extent() const noexcept;
    [[nodiscard]] bool motion_validation_enabled() const noexcept;
    [[nodiscard]] bool read_motion_validation(
        std::uint32_t slot_index,
        MotionValidationSample& sample) noexcept;
    void commit_frame_history() noexcept;
    [[nodiscard]] VkExtent2D output_extent() const noexcept;
    [[nodiscard]] bool record(
        VkCommandBuffer command_buffer,
        std::uint32_t slot_index) noexcept;

    void destroy() noexcept;

private:
    struct HistoryImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        bool initialized = false;
    };

    struct MotionField {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        bool initialized = false;
    };

    struct Slot {
        VkImage source = VK_NULL_HANDLE;
        VkImageView source_view = VK_NULL_HANDLE;
        VkImage output = VK_NULL_HANDLE;
        VkDeviceMemory output_memory = VK_NULL_HANDLE;
        VkImageView output_view = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkImage sharpened_output = VK_NULL_HANDLE;
        VkDeviceMemory sharpened_output_memory = VK_NULL_HANDLE;
        VkImageView sharpened_output_view = VK_NULL_HANDLE;
        VkDescriptorSet sharpen_descriptor_set = VK_NULL_HANDLE;
        VkBuffer motion_validation_buffer = VK_NULL_HANDLE;
        VkDeviceMemory motion_validation_memory = VK_NULL_HANDLE;
        bool motion_validation_written = false;
    };

    [[nodiscard]] std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags preferred_flags) const noexcept;
    [[nodiscard]] bool load_functions(
        PFN_vkGetDeviceProcAddr get_device_proc_addr) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkExtent2D source_extent_{};
    VkExtent2D output_extent_{};
    VkFormat source_format_ = VK_FORMAT_UNDEFINED;
    ScaleFilter filter_ = ScaleFilter::Bilinear;
    float sharpening_strength_ = 0.0F;

    PFN_vkCreateImage create_image_ = nullptr;
    PFN_vkDestroyImage destroy_image_ = nullptr;
    PFN_vkCreateBuffer create_buffer_ = nullptr;
    PFN_vkDestroyBuffer destroy_buffer_ = nullptr;
    PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements_ = nullptr;
    PFN_vkGetImageMemoryRequirements get_image_memory_requirements_ = nullptr;
    PFN_vkAllocateMemory allocate_memory_ = nullptr;
    PFN_vkFreeMemory free_memory_ = nullptr;
    PFN_vkBindImageMemory bind_image_memory_ = nullptr;
    PFN_vkBindBufferMemory bind_buffer_memory_ = nullptr;
    PFN_vkMapMemory map_memory_ = nullptr;
    PFN_vkUnmapMemory unmap_memory_ = nullptr;
    PFN_vkCreateImageView create_image_view_ = nullptr;
    PFN_vkDestroyImageView destroy_image_view_ = nullptr;
    PFN_vkCreateSampler create_sampler_ = nullptr;
    PFN_vkDestroySampler destroy_sampler_ = nullptr;
    PFN_vkCreateDescriptorSetLayout create_descriptor_set_layout_ = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroy_descriptor_set_layout_ = nullptr;
    PFN_vkCreateDescriptorPool create_descriptor_pool_ = nullptr;
    PFN_vkDestroyDescriptorPool destroy_descriptor_pool_ = nullptr;
    PFN_vkAllocateDescriptorSets allocate_descriptor_sets_ = nullptr;
    PFN_vkUpdateDescriptorSets update_descriptor_sets_ = nullptr;
    PFN_vkCreatePipelineLayout create_pipeline_layout_ = nullptr;
    PFN_vkDestroyPipelineLayout destroy_pipeline_layout_ = nullptr;
    PFN_vkCreateShaderModule create_shader_module_ = nullptr;
    PFN_vkDestroyShaderModule destroy_shader_module_ = nullptr;
    PFN_vkCreateComputePipelines create_compute_pipelines_ = nullptr;
    PFN_vkDestroyPipeline destroy_pipeline_ = nullptr;
    PFN_vkCreateQueryPool create_query_pool_ = nullptr;
    PFN_vkDestroyQueryPool destroy_query_pool_ = nullptr;
    PFN_vkGetQueryPoolResults get_query_pool_results_ = nullptr;
    PFN_vkCmdResetQueryPool cmd_reset_query_pool_ = nullptr;
    PFN_vkCmdWriteTimestamp cmd_write_timestamp_ = nullptr;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier_ = nullptr;
    PFN_vkCmdCopyImage cmd_copy_image_ = nullptr;
    PFN_vkCmdCopyImageToBuffer cmd_copy_image_to_buffer_ = nullptr;
    PFN_vkCmdBindPipeline cmd_bind_pipeline_ = nullptr;
    PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets_ = nullptr;
    PFN_vkCmdDispatch cmd_dispatch_ = nullptr;

    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout motion_descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout motion_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipeline sharpen_pipeline_ = VK_NULL_HANDLE;
    VkPipeline motion_pipeline_ = VK_NULL_HANDLE;
    VkQueryPool timing_query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ns_ = 0.0F;
    std::uint32_t timestamp_valid_bits_ = 0;

    std::vector<Slot> slots_;
    std::array<HistoryImage, 2> history_{};
    std::array<MotionField, 2> motion_fields_{};
    VkExtent2D motion_extent_{};
    std::uint32_t history_write_index_ = 0;
    std::uint64_t history_frame_count_ = 0;
    bool motion_validation_enabled_ = false;
    bool ready_ = false;
};

} // namespace ofg::vulkan
