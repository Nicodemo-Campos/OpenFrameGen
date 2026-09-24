// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace ofg::vulkan {

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
        VkExtent2D extent,
        VkFormat source_format,
        const std::vector<VkImage>& source_images) noexcept;

    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool record(
        VkCommandBuffer command_buffer,
        std::uint32_t slot_index) const noexcept;

    void destroy() noexcept;

private:
    struct Slot {
        VkImage source = VK_NULL_HANDLE;
        VkImageView source_view = VK_NULL_HANDLE;
        VkImage output = VK_NULL_HANDLE;
        VkDeviceMemory output_memory = VK_NULL_HANDLE;
        VkImageView output_view = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    };

    [[nodiscard]] std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags preferred_flags) const noexcept;
    [[nodiscard]] bool load_functions(
        PFN_vkGetDeviceProcAddr get_device_proc_addr) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkExtent2D extent_{};
    VkFormat source_format_ = VK_FORMAT_UNDEFINED;

    PFN_vkCreateImage create_image_ = nullptr;
    PFN_vkDestroyImage destroy_image_ = nullptr;
    PFN_vkGetImageMemoryRequirements get_image_memory_requirements_ = nullptr;
    PFN_vkAllocateMemory allocate_memory_ = nullptr;
    PFN_vkFreeMemory free_memory_ = nullptr;
    PFN_vkBindImageMemory bind_image_memory_ = nullptr;
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
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier_ = nullptr;
    PFN_vkCmdBindPipeline cmd_bind_pipeline_ = nullptr;
    PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets_ = nullptr;
    PFN_vkCmdDispatch cmd_dispatch_ = nullptr;

    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    std::vector<Slot> slots_;
    bool ready_ = false;
};

} // namespace ofg::vulkan
