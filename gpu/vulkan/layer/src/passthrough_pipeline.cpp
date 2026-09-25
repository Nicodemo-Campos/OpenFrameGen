// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "passthrough_pipeline.hpp"

#ifndef OFG_VULKAN_PASSTHROUGH_ENABLED
#define OFG_VULKAN_PASSTHROUGH_ENABLED 0
#endif

#if OFG_VULKAN_PASSTHROUGH_ENABLED
#include "motion_estimation_spv.hpp"
#include "passthrough_spv.hpp"
#include "sharpen_spv.hpp"
#endif

#include <array>
#include <cstddef>
#include <vector>

namespace ofg::vulkan {

namespace {

constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kMotionFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr std::uint32_t kMotionBlockSize = 8;
constexpr std::uint32_t kTimingQueriesPerSlot = 3;

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
    OFG_LOAD(vkCreateQueryPool, create_query_pool_);
    OFG_LOAD(vkDestroyQueryPool, destroy_query_pool_);
    OFG_LOAD(vkGetQueryPoolResults, get_query_pool_results_);
    OFG_LOAD(vkCmdResetQueryPool, cmd_reset_query_pool_);
    OFG_LOAD(vkCmdWriteTimestamp, cmd_write_timestamp_);
    OFG_LOAD(vkCmdPipelineBarrier, cmd_pipeline_barrier_);
    OFG_LOAD(vkCmdCopyImage, cmd_copy_image_);
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
           cmd_copy_image_ != nullptr &&
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
    float sharpening_strength,
    float timestamp_period_ns,
    std::uint32_t timestamp_valid_bits,
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
    (void)sharpening_strength;
    (void)timestamp_period_ns;
    (void)timestamp_valid_bits;
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
    sharpening_strength_ = sharpening_strength;
    timestamp_period_ns_ = timestamp_period_ns;
    timestamp_valid_bits_ = timestamp_valid_bits;

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

    if (sharpening_strength_ > 0.0F) {
        const VkShaderModuleCreateInfo sharpen_shader_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = generated::kSharpenSpirvSize,
            .pCode = reinterpret_cast<const std::uint32_t*>(
                generated::kSharpenSpirv),
        };

        VkShaderModule sharpen_shader_module = VK_NULL_HANDLE;
        if (create_shader_module_(
                device_,
                &sharpen_shader_info,
                nullptr,
                &sharpen_shader_module) != VK_SUCCESS) {
            destroy();
            return false;
        }

        const VkSpecializationMapEntry strength_entry{
            .constantID = 0,
            .offset = 0,
            .size = sizeof(sharpening_strength_),
        };

        const VkSpecializationInfo sharpen_specialization{
            .mapEntryCount = 1,
            .pMapEntries = &strength_entry,
            .dataSize = sizeof(sharpening_strength_),
            .pData = &sharpening_strength_,
        };

        const VkPipelineShaderStageCreateInfo sharpen_stage_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sharpen_shader_module,
            .pName = "main",
            .pSpecializationInfo = &sharpen_specialization,
        };

        const VkComputePipelineCreateInfo sharpen_pipeline_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = sharpen_stage_info,
            .layout = pipeline_layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        const VkResult sharpen_pipeline_result =
            create_compute_pipelines_(
                device_,
                VK_NULL_HANDLE,
                1,
                &sharpen_pipeline_info,
                nullptr,
                &sharpen_pipeline_);

        destroy_shader_module_(
            device_,
            sharpen_shader_module,
            nullptr);

        if (sharpen_pipeline_result != VK_SUCCESS) {
            destroy();
            return false;
        }
    }

    const std::array<VkDescriptorSetLayoutBinding, 3>
        motion_bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
        };

    const VkDescriptorSetLayoutCreateInfo motion_set_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount =
            static_cast<std::uint32_t>(motion_bindings.size()),
        .pBindings = motion_bindings.data(),
    };

    if (create_descriptor_set_layout_(
            device_,
            &motion_set_layout_info,
            nullptr,
            &motion_descriptor_set_layout_) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const VkPipelineLayoutCreateInfo motion_pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &motion_descriptor_set_layout_,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };

    if (create_pipeline_layout_(
            device_,
            &motion_pipeline_layout_info,
            nullptr,
            &motion_pipeline_layout_) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const VkShaderModuleCreateInfo motion_shader_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = generated::kMotionEstimationSpirvSize,
        .pCode = reinterpret_cast<const std::uint32_t*>(
            generated::kMotionEstimationSpirv),
    };

    VkShaderModule motion_shader_module = VK_NULL_HANDLE;
    if (create_shader_module_(
            device_,
            &motion_shader_info,
            nullptr,
            &motion_shader_module) != VK_SUCCESS) {
        destroy();
        return false;
    }

    const VkPipelineShaderStageCreateInfo motion_stage_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = motion_shader_module,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };

    const VkComputePipelineCreateInfo motion_pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = motion_stage_info,
        .layout = motion_pipeline_layout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    const VkResult motion_pipeline_result =
        create_compute_pipelines_(
            device_,
            VK_NULL_HANDLE,
            1,
            &motion_pipeline_info,
            nullptr,
            &motion_pipeline_);

    destroy_shader_module_(
        device_,
        motion_shader_module,
        nullptr);

    if (motion_pipeline_result != VK_SUCCESS) {
        destroy();
        return false;
    }

    motion_extent_ = VkExtent2D{
        (output_extent_.width + kMotionBlockSize - 1u) /
            kMotionBlockSize,
        (output_extent_.height + kMotionBlockSize - 1u) /
            kMotionBlockSize,
    };

    const std::uint32_t slot_count =
        static_cast<std::uint32_t>(source_images.size());

    const bool timing_functions_available =
        create_query_pool_ != nullptr &&
        destroy_query_pool_ != nullptr &&
        get_query_pool_results_ != nullptr &&
        cmd_reset_query_pool_ != nullptr &&
        cmd_write_timestamp_ != nullptr;

    if (timing_functions_available &&
        timestamp_period_ns_ > 0.0F &&
        timestamp_valid_bits_ > 0) {
        const VkQueryPoolCreateInfo query_pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = slot_count * kTimingQueriesPerSlot,
            .pipelineStatistics = 0,
        };

        if (create_query_pool_(
                device_,
                &query_pool_info,
                nullptr,
                &timing_query_pool_) != VK_SUCCESS) {
            timing_query_pool_ = VK_NULL_HANDLE;
        }
    }

    const std::uint32_t descriptor_multiplier =
        sharpening_strength_ > 0.0F ? 2u : 1u;

    const std::array<VkDescriptorPoolSize, 2> pool_sizes{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount =
                slot_count * descriptor_multiplier + 4u,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount =
                slot_count * descriptor_multiplier + 2u,
        },
    };

    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = slot_count * descriptor_multiplier + 2u,
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

    const std::uint32_t descriptor_set_count =
        slot_count * descriptor_multiplier;
    std::vector<VkDescriptorSetLayout> layouts(
        descriptor_set_count,
        descriptor_set_layout_);
    std::vector<VkDescriptorSet> descriptor_sets(descriptor_set_count);

    const VkDescriptorSetAllocateInfo descriptor_allocate_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount = descriptor_set_count,
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
        if (sharpening_strength_ > 0.0F) {
            slot.sharpen_descriptor_set =
                descriptor_sets[slot_count + index];
        }

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
                VK_IMAGE_USAGE_SAMPLED_BIT |
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

        if (sharpening_strength_ > 0.0F) {
            const VkImageCreateInfo sharpen_output_info{
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
                    &sharpen_output_info,
                    nullptr,
                    &slot.sharpened_output) != VK_SUCCESS) {
                destroy();
                return false;
            }

            VkMemoryRequirements sharpen_requirements{};
            get_image_memory_requirements_(
                device_,
                slot.sharpened_output,
                &sharpen_requirements);

            const std::uint32_t sharpen_memory_type =
                find_memory_type(
                    sharpen_requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            if (sharpen_memory_type == UINT32_MAX) {
                destroy();
                return false;
            }

            const VkMemoryAllocateInfo sharpen_allocation_info{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = sharpen_requirements.size,
                .memoryTypeIndex = sharpen_memory_type,
            };

            if (allocate_memory_(
                    device_,
                    &sharpen_allocation_info,
                    nullptr,
                    &slot.sharpened_output_memory) != VK_SUCCESS) {
                destroy();
                return false;
            }

            if (bind_image_memory_(
                    device_,
                    slot.sharpened_output,
                    slot.sharpened_output_memory,
                    0) != VK_SUCCESS) {
                destroy();
                return false;
            }

            const VkImageViewCreateInfo sharpen_output_view_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = slot.sharpened_output,
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
                    &sharpen_output_view_info,
                    nullptr,
                    &slot.sharpened_output_view) != VK_SUCCESS) {
                destroy();
                return false;
            }
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

        if (sharpening_strength_ > 0.0F) {
            const VkDescriptorImageInfo sharpen_source_descriptor{
                .sampler = sampler_,
                .imageView = slot.output_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            const VkDescriptorImageInfo sharpen_output_descriptor{
                .sampler = VK_NULL_HANDLE,
                .imageView = slot.sharpened_output_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };

            const std::array<VkWriteDescriptorSet, 2> sharpen_writes{
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = slot.sharpen_descriptor_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType =
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &sharpen_source_descriptor,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = slot.sharpen_descriptor_set,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .pImageInfo = &sharpen_output_descriptor,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                },
            };

            update_descriptor_sets_(
                device_,
                static_cast<std::uint32_t>(sharpen_writes.size()),
                sharpen_writes.data(),
                0,
                nullptr);
        }
    }

    for (auto& history_image : history_) {
        const VkImageCreateInfo history_info{
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
                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (create_image_(
                device_,
                &history_info,
                nullptr,
                &history_image.image) != VK_SUCCESS) {
            destroy();
            return false;
        }

        VkMemoryRequirements history_requirements{};
        get_image_memory_requirements_(
            device_,
            history_image.image,
            &history_requirements);

        const std::uint32_t history_memory_type =
            find_memory_type(
                history_requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (history_memory_type == UINT32_MAX) {
            destroy();
            return false;
        }

        const VkMemoryAllocateInfo history_allocation_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = history_requirements.size,
            .memoryTypeIndex = history_memory_type,
        };

        if (allocate_memory_(
                device_,
                &history_allocation_info,
                nullptr,
                &history_image.memory) != VK_SUCCESS) {
            destroy();
            return false;
        }

        if (bind_image_memory_(
                device_,
                history_image.image,
                history_image.memory,
                0) != VK_SUCCESS) {
            destroy();
            return false;
        }

        const VkImageViewCreateInfo history_view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = history_image.image,
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
                &history_view_info,
                nullptr,
                &history_image.view) != VK_SUCCESS) {
            destroy();
            return false;
        }
    }

    for (auto& motion_field : motion_fields_) {
        const VkImageCreateInfo motion_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = kMotionFormat,
            .extent = VkExtent3D{
                motion_extent_.width,
                motion_extent_.height,
                1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage =
                VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (create_image_(
                device_,
                &motion_info,
                nullptr,
                &motion_field.image) != VK_SUCCESS) {
            destroy();
            return false;
        }

        VkMemoryRequirements motion_requirements{};
        get_image_memory_requirements_(
            device_,
            motion_field.image,
            &motion_requirements);

        const std::uint32_t motion_memory_type =
            find_memory_type(
                motion_requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (motion_memory_type == UINT32_MAX) {
            destroy();
            return false;
        }

        const VkMemoryAllocateInfo motion_allocation_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = motion_requirements.size,
            .memoryTypeIndex = motion_memory_type,
        };

        if (allocate_memory_(
                device_,
                &motion_allocation_info,
                nullptr,
                &motion_field.memory) != VK_SUCCESS) {
            destroy();
            return false;
        }

        if (bind_image_memory_(
                device_,
                motion_field.image,
                motion_field.memory,
                0) != VK_SUCCESS) {
            destroy();
            return false;
        }

        const VkImageViewCreateInfo motion_view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = motion_field.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = kMotionFormat,
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
                &motion_view_info,
                nullptr,
                &motion_field.view) != VK_SUCCESS) {
            destroy();
            return false;
        }
    }

    const std::array<VkDescriptorSetLayout, 2> motion_layouts{
        motion_descriptor_set_layout_,
        motion_descriptor_set_layout_,
    };
    std::array<VkDescriptorSet, 2> motion_descriptor_sets{};

    const VkDescriptorSetAllocateInfo motion_descriptor_allocate_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount =
            static_cast<std::uint32_t>(motion_descriptor_sets.size()),
        .pSetLayouts = motion_layouts.data(),
    };

    if (allocate_descriptor_sets_(
            device_,
            &motion_descriptor_allocate_info,
            motion_descriptor_sets.data()) != VK_SUCCESS) {
        destroy();
        return false;
    }

    for (std::uint32_t index = 0;
         index < motion_fields_.size();
         ++index) {
        const std::uint32_t previous_index =
            (index + 1u) %
            static_cast<std::uint32_t>(history_.size());

        auto& motion_field = motion_fields_[index];
        motion_field.descriptor_set =
            motion_descriptor_sets[index];

        const VkDescriptorImageInfo previous_descriptor{
            .sampler = sampler_,
            .imageView = history_[previous_index].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const VkDescriptorImageInfo current_descriptor{
            .sampler = sampler_,
            .imageView = history_[index].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const VkDescriptorImageInfo motion_output_descriptor{
            .sampler = VK_NULL_HANDLE,
            .imageView = motion_field.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        const std::array<VkWriteDescriptorSet, 3> motion_writes{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = motion_field.descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &previous_descriptor,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = motion_field.descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &current_descriptor,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = motion_field.descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &motion_output_descriptor,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
        };

        update_descriptor_sets_(
            device_,
            static_cast<std::uint32_t>(motion_writes.size()),
            motion_writes.data(),
            0,
            nullptr);
    }

    history_write_index_ = 0;
    history_frame_count_ = 0;

    ready_ = true;
    return true;
#endif
}

bool VulkanPassthroughPipeline::ready() const noexcept {
    return ready_;
}

bool VulkanPassthroughPipeline::sharpening_enabled() const noexcept {
    return sharpen_pipeline_ != VK_NULL_HANDLE &&
           sharpening_strength_ > 0.0F;
}

float VulkanPassthroughPipeline::sharpening_strength() const noexcept {
    return sharpening_strength_;
}

bool VulkanPassthroughPipeline::timing_enabled() const noexcept {
    return timing_query_pool_ != VK_NULL_HANDLE &&
           get_query_pool_results_ != nullptr &&
           cmd_reset_query_pool_ != nullptr &&
           cmd_write_timestamp_ != nullptr &&
           timestamp_period_ns_ > 0.0F &&
           timestamp_valid_bits_ > 0;
}

bool VulkanPassthroughPipeline::read_timing(
    std::uint32_t slot_index,
    GpuTimingSample& sample) const noexcept {
    sample = {};

    if (!timing_enabled() ||
        slot_index >= slots_.size()) {
        return false;
    }

    std::array<std::uint64_t, kTimingQueriesPerSlot> timestamps{};
    const std::uint32_t first_query =
        slot_index * kTimingQueriesPerSlot;

    const VkResult result =
        get_query_pool_results_(
            device_,
            timing_query_pool_,
            first_query,
            kTimingQueriesPerSlot,
            sizeof(timestamps),
            timestamps.data(),
            sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT);

    if (result != VK_SUCCESS) {
        return false;
    }

    const auto tick_delta =
        [this](std::uint64_t begin, std::uint64_t end) noexcept {
            if (timestamp_valid_bits_ >= 64) {
                return end - begin;
            }

            const std::uint64_t mask =
                (std::uint64_t{1} << timestamp_valid_bits_) - 1;
            return (end - begin) & mask;
        };

    const std::uint64_t scaler_ticks =
        tick_delta(timestamps[0], timestamps[1]);
    const std::uint64_t sharpen_ticks =
        sharpening_enabled()
            ? tick_delta(timestamps[1], timestamps[2])
            : 0;
    const std::uint64_t total_ticks =
        sharpening_enabled()
            ? tick_delta(timestamps[0], timestamps[2])
            : scaler_ticks;

    constexpr double nanoseconds_per_millisecond = 1'000'000.0;
    const double period =
        static_cast<double>(timestamp_period_ns_);

    sample.scaler_ms =
        static_cast<double>(scaler_ticks) *
        period /
        nanoseconds_per_millisecond;
    sample.sharpening_ms =
        static_cast<double>(sharpen_ticks) *
        period /
        nanoseconds_per_millisecond;
    sample.total_ms =
        static_cast<double>(total_ticks) *
        period /
        nanoseconds_per_millisecond;

    return true;
}

bool VulkanPassthroughPipeline::frame_history_ready() const noexcept {
    return history_frame_count_ >= history_.size();
}

std::uint64_t VulkanPassthroughPipeline::history_frame_count() const noexcept {
    return history_frame_count_;
}

bool VulkanPassthroughPipeline::motion_estimation_enabled() const noexcept {
    return motion_pipeline_ != VK_NULL_HANDLE &&
           motion_pipeline_layout_ != VK_NULL_HANDLE &&
           motion_descriptor_set_layout_ != VK_NULL_HANDLE;
}

bool VulkanPassthroughPipeline::motion_field_ready() const noexcept {
    return motion_estimation_enabled() &&
           history_frame_count_ >= 2;
}

VkExtent2D VulkanPassthroughPipeline::motion_field_extent() const noexcept {
    return motion_extent_;
}

void VulkanPassthroughPipeline::commit_frame_history() noexcept {
    if (!ready_ ||
        history_write_index_ >= history_.size() ||
        history_[history_write_index_].image == VK_NULL_HANDLE) {
        return;
    }

    const bool recorded_motion =
        history_frame_count_ >= 1 &&
        history_write_index_ < motion_fields_.size();

    history_[history_write_index_].initialized = true;

    if (recorded_motion) {
        motion_fields_[history_write_index_].initialized = true;
    }

    ++history_frame_count_;
    history_write_index_ =
        (history_write_index_ + 1u) %
        static_cast<std::uint32_t>(history_.size());
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

    const bool record_timing = timing_enabled();
    const std::uint32_t first_timing_query =
        slot_index * kTimingQueriesPerSlot;

    if (record_timing) {
        cmd_reset_query_pool_(
            command_buffer,
            timing_query_pool_,
            first_timing_query,
            kTimingQueriesPerSlot);
    }

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

    if (record_timing) {
        cmd_write_timestamp_(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            timing_query_pool_,
            first_timing_query);
    }

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

    if (record_timing) {
        cmd_write_timestamp_(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            timing_query_pool_,
            first_timing_query + 1);
    }

    if (sharpening_enabled()) {
        std::array<VkImageMemoryBarrier, 2> sharpen_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
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
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = slot.sharpened_output,
                .subresourceRange = VkImageSubresourceRange{
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0,
                    1,
                    0,
                    1,
                },
            },
        };

        cmd_pipeline_barrier_(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            static_cast<std::uint32_t>(sharpen_barriers.size()),
            sharpen_barriers.data());

        cmd_bind_pipeline_(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            sharpen_pipeline_);

        cmd_bind_descriptor_sets_(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout_,
            0,
            1,
            &slot.sharpen_descriptor_set,
            0,
            nullptr);

        cmd_dispatch_(
            command_buffer,
            group_count_x,
            group_count_y,
            1);
    }

    if (record_timing) {
        cmd_write_timestamp_(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            timing_query_pool_,
            first_timing_query + 2);
    }

    if (history_write_index_ >= history_.size()) {
        return false;
    }

    const auto& history_target =
        history_[history_write_index_];
    const VkImage final_output =
        sharpening_enabled()
            ? slot.sharpened_output
            : slot.output;

    if (final_output == VK_NULL_HANDLE ||
        history_target.image == VK_NULL_HANDLE) {
        return false;
    }

    std::array<VkImageMemoryBarrier, 2> history_prepare{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = final_output,
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
            .srcAccessMask =
                history_target.initialized
                    ? VK_ACCESS_SHADER_READ_BIT
                    : 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout =
                history_target.initialized
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = history_target.image,
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        },
    };

    cmd_pipeline_barrier_(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        static_cast<std::uint32_t>(history_prepare.size()),
        history_prepare.data());

    const VkImageCopy history_copy{
        .srcSubresource = VkImageSubresourceLayers{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            0,
            1,
        },
        .srcOffset = VkOffset3D{0, 0, 0},
        .dstSubresource = VkImageSubresourceLayers{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            0,
            1,
        },
        .dstOffset = VkOffset3D{0, 0, 0},
        .extent = VkExtent3D{
            output_extent_.width,
            output_extent_.height,
            1,
        },
    };

    cmd_copy_image_(
        command_buffer,
        final_output,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        history_target.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &history_copy);

    const VkImageMemoryBarrier history_ready{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = history_target.image,
        .subresourceRange = VkImageSubresourceRange{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        },
    };

    cmd_pipeline_barrier_(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &history_ready);

    const bool can_estimate_motion =
        motion_estimation_enabled() &&
        history_frame_count_ >= 1 &&
        history_write_index_ < motion_fields_.size();

    if (can_estimate_motion) {
        const std::uint32_t previous_history_index =
            (history_write_index_ + 1u) %
            static_cast<std::uint32_t>(history_.size());

        if (!history_[previous_history_index].initialized) {
            return false;
        }

        const auto& motion_target =
            motion_fields_[history_write_index_];

        if (motion_target.image == VK_NULL_HANDLE ||
            motion_target.descriptor_set == VK_NULL_HANDLE) {
            return false;
        }

        const VkImageMemoryBarrier prepare_motion{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask =
                motion_target.initialized
                    ? VK_ACCESS_SHADER_READ_BIT
                    : 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout =
                motion_target.initialized
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = motion_target.image,
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        };

        cmd_pipeline_barrier_(
            command_buffer,
            motion_target.initialized
                ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &prepare_motion);

        cmd_bind_pipeline_(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            motion_pipeline_);

        cmd_bind_descriptor_sets_(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            motion_pipeline_layout_,
            0,
            1,
            &motion_target.descriptor_set,
            0,
            nullptr);

        const std::uint32_t motion_group_count_x =
            (motion_extent_.width + 7u) / 8u;
        const std::uint32_t motion_group_count_y =
            (motion_extent_.height + 7u) / 8u;

        cmd_dispatch_(
            command_buffer,
            motion_group_count_x,
            motion_group_count_y,
            1);

        const VkImageMemoryBarrier motion_ready{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = motion_target.image,
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        };

        cmd_pipeline_barrier_(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &motion_ready);
    }

    return true;
}

void VulkanPassthroughPipeline::destroy() noexcept {
    ready_ = false;

    if (device_ != VK_NULL_HANDLE &&
        destroy_query_pool_ != nullptr &&
        timing_query_pool_ != VK_NULL_HANDLE) {
        destroy_query_pool_(
            device_,
            timing_query_pool_,
            nullptr);
    }
    timing_query_pool_ = VK_NULL_HANDLE;

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
        motion_pipeline_ != VK_NULL_HANDLE) {
        destroy_pipeline_(device_, motion_pipeline_, nullptr);
    }
    motion_pipeline_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_pipeline_ != nullptr &&
        sharpen_pipeline_ != VK_NULL_HANDLE) {
        destroy_pipeline_(device_, sharpen_pipeline_, nullptr);
    }
    sharpen_pipeline_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_pipeline_ != nullptr &&
        pipeline_ != VK_NULL_HANDLE) {
        destroy_pipeline_(device_, pipeline_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;

    if (device_ != VK_NULL_HANDLE &&
        destroy_pipeline_layout_ != nullptr &&
        motion_pipeline_layout_ != VK_NULL_HANDLE) {
        destroy_pipeline_layout_(
            device_,
            motion_pipeline_layout_,
            nullptr);
    }
    motion_pipeline_layout_ = VK_NULL_HANDLE;

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
        motion_descriptor_set_layout_ != VK_NULL_HANDLE) {
        destroy_descriptor_set_layout_(
            device_,
            motion_descriptor_set_layout_,
            nullptr);
    }
    motion_descriptor_set_layout_ = VK_NULL_HANDLE;

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

    for (auto& motion_field : motion_fields_) {
        if (device_ != VK_NULL_HANDLE &&
            destroy_image_view_ != nullptr &&
            motion_field.view != VK_NULL_HANDLE) {
            destroy_image_view_(
                device_,
                motion_field.view,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            destroy_image_ != nullptr &&
            motion_field.image != VK_NULL_HANDLE) {
            destroy_image_(
                device_,
                motion_field.image,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            free_memory_ != nullptr &&
            motion_field.memory != VK_NULL_HANDLE) {
            free_memory_(
                device_,
                motion_field.memory,
                nullptr);
        }
    }
    motion_fields_ = {};
    motion_extent_ = {};

    for (auto& history_image : history_) {
        if (device_ != VK_NULL_HANDLE &&
            destroy_image_view_ != nullptr &&
            history_image.view != VK_NULL_HANDLE) {
            destroy_image_view_(
                device_,
                history_image.view,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            destroy_image_ != nullptr &&
            history_image.image != VK_NULL_HANDLE) {
            destroy_image_(
                device_,
                history_image.image,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            free_memory_ != nullptr &&
            history_image.memory != VK_NULL_HANDLE) {
            free_memory_(
                device_,
                history_image.memory,
                nullptr);
        }
    }
    history_ = {};
    history_write_index_ = 0;
    history_frame_count_ = 0;

    for (auto& slot : slots_) {
        if (device_ != VK_NULL_HANDLE &&
            destroy_image_view_ != nullptr &&
            slot.sharpened_output_view != VK_NULL_HANDLE) {
            destroy_image_view_(
                device_,
                slot.sharpened_output_view,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            destroy_image_ != nullptr &&
            slot.sharpened_output != VK_NULL_HANDLE) {
            destroy_image_(
                device_,
                slot.sharpened_output,
                nullptr);
        }

        if (device_ != VK_NULL_HANDLE &&
            free_memory_ != nullptr &&
            slot.sharpened_output_memory != VK_NULL_HANDLE) {
            free_memory_(
                device_,
                slot.sharpened_output_memory,
                nullptr);
        }

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
    sharpening_strength_ = 0.0F;
    timestamp_period_ns_ = 0.0F;
    timestamp_valid_bits_ = 0;
}

} // namespace ofg::vulkan
