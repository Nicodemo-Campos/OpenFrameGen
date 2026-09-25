// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "passthrough_pipeline.hpp"

#ifndef OFG_VULKAN_PASSTHROUGH_ENABLED
#define OFG_VULKAN_PASSTHROUGH_ENABLED 0
#endif

#if OFG_VULKAN_PASSTHROUGH_ENABLED
#include "passthrough_spv.hpp"
#endif

#include <array>
#include <cstddef>
#include <vector>

namespace ofg::vulkan {

namespace {

constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;

template <typename Function>
[[nodiscard]] Function load_device_function(
    PFN_vkGetDeviceProcAddr get_device_proc_addr,
    VkDevice device,
    const char* name) noexcept {
    if (get_device_proc_addr == nullptr || device == VK_NULL_HANDLE) {
        return nullptr;
    }

    return reinterpret_cast<Function>(
        get_device_proc_addr(device, name));
}

} // namespace

VulkanPassthroughPipeline::~VulkanPassthroughPipeline() {
    destroy();
}

bool VulkanPassthroughPipeline::build_available() noexcept {
#if OFG_VULKAN_PASSTHROUGH_ENABLED
    return true;
#else
    return false;
#endif
}

bool VulkanPassthroughPipeline::supports_source_format(
    VkFormat format) noexcept {
    return format == VK_FORMAT_B8G8R8A8_UNORM ||
           format == VK_FORMAT_R8G8B8A8_UNORM;
}

bool VulkanPassthroughPipeline::load_functions(
    PFN_vkGetDeviceProcAddr get_device_proc_addr) noexcept {
#define OFG_LOAD(name, field) \
    field = load_device_function<PFN_##name>( \
        get_device_proc_addr, device_, #name)

    OFG_LOAD(vkCreateImage, create_image_);
    OFG_LOAD(vkDestroyImage, destroy_image_);
    OFG_LOAD(vkGetImageMemoryRequirements, get_image_memory_requirements_);
    OFG_LOAD(vkAllocateMemory, allocate_memory_);
    OFG_LOAD(vkFreeMemory, free_memory_);
    OFG_LOAD(vkBindImageMemory, bind_image_memory_);
    OFG_LOAD(vkCreateImageView, create_image_view_);
    OFG_LOAD(vkDestroyImageView, destroy_image_view_);
    OFG_LOAD(vkCreateSampler, create_sampler_);
    OFG_LOAD(vkDestroySampler, destroy_sampler_);
    OFG_LOAD(vkCreateDescriptorSetLayout, create_descriptor_set_layout_);
    OFG_LOAD(vkDestroyDescriptorSetLayout, destroy_descriptor_set_layout_);
    OFG_LOAD(vkCreateDescriptorPool, create_descriptor_pool_);
    OFG_LOAD(vkDestroyDescriptorPool, destroy_descriptor_pool_);
    OFG_LOAD(vkAllocateDescriptorSets, allocate_descriptor_sets_);
    OFG_LOAD(vkUpdateDescriptorSets, update_descriptor_sets_);
    OFG_LOAD(vkCreatePipelineLayout, create_pipeline_layout_);
    OFG_LOAD(vkDestroyPipelineLayout, destroy_pipeline_layout_);
    OFG_LOAD(vkCreateShaderModule, create_shader_module_);
    OFG_LOAD(vkDestroyShaderModule, destroy_shader_module_);
    OFG_LOAD(vkCreateComputePipelines, create_compute_pipelines_);
    OFG_LOAD(vkDestroyPipeline, destroy_pipeline_);
    OFG_LOAD(vkCmdPipelineBarrier, cmd_pipeline_barrier_);
    OFG_LOAD(vkCmdBindPipeline, cmd_bind_pipeline_);
    OFG_LOAD(vkCmdBindDescriptorSets, cmd_bind_descriptor_sets_);
    OFG_LOAD(vkCmdDispatch, cmd_dispatch_);

#undef OFG_LOAD

    return create_image_ != nullptr &&
           destroy_image_ != nullptr &&
           get_image_memory_requirements_ != nullptr &&
           allocate_memory_ != nullptr &&
           free_memory_ != nullptr &&
           bind_image_memory_ != nullptr &&
           create_image_view_ != nullptr &&
           destroy_image_view_ != nullptr &&
           create_sampler_ != nullptr &&
           destroy_sampler_ != nullptr &&
           create_descriptor_set_layout_ != nullptr &&
           destroy_descriptor_set_layout_ != nullptr &&
           create_descriptor_pool_ != nullptr &&
           destroy_descriptor_pool_ != nullptr &&
           allocate_descriptor_sets_ != nullptr &&
           update_descriptor_sets_ != nullptr &&
           create_pipeline_layout_ != nullptr &&
           destroy_pipeline_layout_ != nullptr &&
           create_shader_module_ != nullptr &&
           destroy_shader_module_ != nullptr &&
           create_compute_pipelines_ != nullptr &&
           destroy_pipeline_ != nullptr &&
           cmd_pipeline_barrier_ != nullptr &&
           cmd_bind_pipeline_ != nullptr &&
           cmd_bind_descriptor_sets_ != nullptr &&
           cmd_dispatch_ != nullptr;
}

std::uint32_t VulkanPassthroughPipeline::find_memory_type(
    std::uint32_t type_bits,
    VkMemoryPropertyFlags preferred_flags) const noexcept {
    for (std::uint32_t index = 0;
         index < memory_properties_.memoryTypeCount;
         ++index) {
        const bool compatible = (type_bits & (1u << index)) != 0;
        const bool preferred =
            (memory_properties_.memoryTypes[index].propertyFlags &
             preferred_flags) == preferred_flags;

        if (compatible && preferred) {
            return index;
        }
    }

    for (std::uint32_t index = 0;
         index < memory_properties_.memoryTypeCount;
         ++index) {
        if ((type_bits & (1u << index)) != 0) {
            return index;
        }
    }

    return UINT32_MAX;
}

bool VulkanPassthroughPipeline::initialize(
    VkDevice device,
    PFN_vkGetDeviceProcAddr get_device_proc_addr,
    const VkPhysicalDeviceMemoryProperties& memory_properties,
    VkExtent2D source_extent,
    VkExtent2D output_extent,
    VkFormat source_format,
    ScaleFilter filter,
    const std::vector<VkImage>& source_images) noexcept {
    destroy();

#if !OFG_VULKAN_PASSTHROUGH_ENABLED
    (void)device;
    (void)get_device_proc_addr;
    (void)memory_properties;
    (void)source_extent;
    (void)output_extent;
    (void)source_format;
    (void)filter;
    (void)source_images;
    return false;
#else
    if (device == VK_NULL_HANDLE ||
        !supports_source_format(source_format) ||
        source_extent.width == 0 ||
        source_extent.height == 0 ||
        output_extent.width == 0 ||
        output_extent.height == 0 ||
        source_images.empty()) {
        return false;
    }

    device_ = device;
    memory_properties_ = memory_properties;
    source_extent_ = source_extent;
    output_extent_ = output_extent;
    source_format_ = source_format;
    filter_ = filter;

    if (!load_functions(get_device_proc_addr)) {
        destroy();
        return false;
    }

    const VkSamplerCreateInfo sampler_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter =
            filter_ == ScaleFilter::Bilinear
                ? VK_FILTER_LINEAR
                : VK_FILTER_NEAREST,
        .minFilter =
            filter_ == ScaleFilter::Bilinear
                ? VK_FILTER_LINEAR
                : VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0F,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0F,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0F,
        .maxLod = 0.0F,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    if (create_sampler_(
            device_,
            &sampler_info,
            nullptr,
            &sampler_) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
    };

    const VkDescriptorSetLayoutCreateInfo set_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    if (create_descriptor_set_layout_(
            device_,
            &set_layout_info,
            nullptr,
            &descriptor_set_layout_) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout_,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };

    if (create_pipeline_layout_(
            device_,
            &pipeline_layout_info,
            nullptr,
            &pipeline_layout_) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const VkShaderModuleCreateInfo shader_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = generated::kPassthroughSpirvSize,
        .pCode = reinterpret_cast<const std::uint32_t*>(
            generated::kPassthroughSpirv),
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (create_shader_module_(
            device_,
            &shader_info,
            nullptr,
            &shader_module) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const std::uint32_t filter_mode =
        static_cast<std::uint32_t>(filter_);

    const VkSpecializationMapEntry filter_entry{
        .constantID = 0,
        .offset = 0,
        .size = sizeof(filter_mode),
    };

    const VkSpecializationInfo specialization_info{
        .mapEntryCount = 1,
        .pMapEntries = &filter_entry,
        .dataSize = sizeof(filter_mode),
        .pData = &filter_mode,
    };

    const VkPipelineShaderStageCreateInfo stage_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = &specialization_info,
    };

    const VkComputePipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage_info,
        .layout = pipeline_layout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    const VkResult pipeline_result =
        create_compute_pipelines_(
            device_,
            VK_NULL_HANDLE,
            1,
            &pipeline_info,
            nullptr,
            &pipeline_);

    destroy_shader_module_(device_, shader_module, nullptr);

    if (pipeline_result != VK_SUCCESS) {
        destroy();
        return false;
    }

    const std::uint32_t slot_count =
        static_cast<std::uint32_t>(source_images.size());

    const std::array<VkDescriptorPoolSize, 2> pool_sizes{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = slot_count,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = slot_count,
        },
    };

    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = slot_count,
        .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };

    if (create_descriptor_pool_(
            device_,
            &pool_info,
            nullptr,
            &descriptor_pool_) != VK_SUCCESS) {
        destroy();
        return false;
    }

    std::vector<VkDescriptorSetLayout> layouts(
        slot_count,
        descriptor_set_layout_);
    std::vector<VkDescriptorSet> descriptor_sets(slot_count);

    const VkDescriptorSetAllocateInfo descriptor_allocate_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount = slot_count,
        .pSetLayouts = layouts.data(),
    };

    if (allocate_descriptor_sets_(
            device_,
            &descriptor_allocate_info,
            descriptor_sets.data()) != VK_SUCCESS) {
        destroy();
        return false;
    }

    slots_.resize(source_images.size());

    for (std::size_t index = 0;
         index < source_images.size();
         ++index) {
        auto& slot = slots_[index];
        slot.source = source_images[index];
        slot.descriptor_set = descriptor_sets[index];

        const VkImageViewCreateInfo source_view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = source_images[index],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = source_format_,
            .components = VkComponentMapping{
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        };

        if (create_image_view_(
                device_,
                &source_view_info,
                nullptr,
                &slot.source_view) != VK_SUCCESS) {
            destroy();
            return false;
        }

        const VkImageCreateInfo output_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = kOutputFormat,
            .extent = VkExtent3D{
                output_extent_.width,
                output_extent_.height,
                1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage =
                VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (create_image_(
                device_,
                &output_info,
                nullptr,
                &slot.output) != VK_SUCCESS) {
            destroy();
            return false;
        }

        VkMemoryRequirements requirements{};
        get_image_memory_requirements_(
            device_,
            slot.output,
            &requirements);

        const std::uint32_t memory_type =
            find_memory_type(
                requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (memory_type == UINT32_MAX) {
            destroy();
            return false;
        }

        const VkMemoryAllocateInfo allocation_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = memory_type,
        };

        if (allocate_memory_(
                device_,
                &allocation_info,
                nullptr,
                &slot.output_memory) != VK_SUCCESS) {
            destroy();
            return false;
        }

        if (bind_image_memory_(
                device_,
                slot.output,
                slot.output_memory,
                0) != VK_SUCCESS) {
            destroy();
            return false;
        }

        const VkImageViewCreateInfo output_view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = slot.output,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = kOutputFormat,
            .components = VkComponentMapping{
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        };

        if (create_image_view_(
                device_,
                &output_view_info,
                nullptr,
                &slot.output_view) != VK_SUCCESS) {
            destroy();
            return false;
        }

        const VkDescriptorImageInfo source_descriptor{
            .sampler = sampler_,
            .imageView = slot.source_view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const VkDescriptorImageInfo output_descriptor{
            .sampler = VK_NULL_HANDLE,
            .imageView = slot.output_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        const std::array<VkWriteDescriptorSet, 2> writes{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = slot.descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &source_descriptor,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = slot.descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &output_descriptor,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
        };

        update_descriptor_sets_(
            device_,
            static_cast<std::uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);
    }

    ready_ = true;
    return true;
#endif
}

bool VulkanPassthroughPipeline::ready() const noexcept {
    return ready_;
}

VkExtent2D VulkanPassthroughPipeline::output_extent() const noexcept {
    return output_extent_;
}

bool VulkanPassthroughPipeline::record(
    VkCommandBuffer command_buffer,
    std::uint32_t slot_index) const noexcept {
    if (!ready_ ||
        command_buffer == VK_NULL_HANDLE ||
        slot_index >= slots_.size()) {
        return false;
    }

    const auto& slot = slots_[slot_index];

    std::array<VkImageMemoryBarrier, 2> barriers{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = VK_NULL_HANDLE,
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = slot.output,
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        },
    };

    barriers[0].image = slot.source;

    cmd_pipeline_barrier_(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        static_cast<std::uint32_t>(barriers.size()),
        barriers.data());

    cmd_bind_pipeline_(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_);

    cmd_bind_descriptor_sets_(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout_,
        0,
        1,
        &slot.descriptor_set,
        0,
        nullptr);

    const std::uint32_t group_count_x =
        (output_extent_.width + 7u) / 8u;
    const std::uint32_t group_count_y =
        (output_extent_.height + 7u) / 8u;

    cmd_dispatch_(
        command_buffer,
        group_count_x,
        group_count_y,
        1);

    return true;
}

void VulkanPassthroughPipeline::destroy() noexcept {
    ready_ = false;

    if (device_ != VK_NULL_HANDLE &&
        destroy_descriptor_pool_ != nullptr &&
        descriptor_pool_ != VK_NULL_HANDLE) {
        destroy_descriptor_pool_(
            device_,
            descriptor_pool_,
            nullptr);
    }
    descriptor_pool_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_pipeline_ != nullptr &&
        pipeline_ != VK_NULL_HANDLE) {
        destroy_pipeline_(device_, pipeline_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_pipeline_layout_ != nullptr &&
        pipeline_layout_ != VK_NULL_HANDLE) {
        destroy_pipeline_layout_(
            device_,
            pipeline_layout_,
            nullptr);
    }
    pipeline_layout_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_descriptor_set_layout_ != nullptr &&
        descriptor_set_layout_ != VK_NULL_HANDLE) {
        destroy_descriptor_set_layout_(
            device_,
            descriptor_set_layout_,
            nullptr);
    }
    descriptor_set_layout_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_sampler_ != nullptr &&
        sampler_ != VK_NULL_HANDLE) {
        destroy_sampler_(device_, sampler_, nullptr);
    }
    sampler_ = VK_NULL_HANDLE;

    for (auto& slot : slots_) {
        if (device_ != VK_NULL_HANDLE &&
            destroy_image_view_ != nullptr &&
            slot.output_view != VK_NULL_HANDLE) {
            destroy_image_view_(
                device_,
                slot.output_view,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            destroy_image_ != nullptr &&
            slot.output != VK_NULL_HANDLE) {
            destroy_image_(device_, slot.output, nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            free_memory_ != nullptr &&
            slot.output_memory != VK_NULL_HANDLE) {
            free_memory_(
                device_,
                slot.output_memory,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            destroy_image_view_ != nullptr &&
            slot.source_view != VK_NULL_HANDLE) {
            destroy_image_view_(
                device_,
                slot.source_view,
                nullptr);
        }
    }

    slots_.clear();
    device_ = VK_NULL_HANDLE;
    memory_properties_ = {};
    source_extent_ = {};
    output_extent_ = {};
    source_format_ = VK_FORMAT_UNDEFINED;
    filter_ = ScaleFilter::Bilinear;
}

} // namespace ofg::vulkan
