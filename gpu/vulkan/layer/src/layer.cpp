// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <openframegen/core/frame_cadence.hpp>

#include "passthrough_pipeline.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct InstanceDispatch {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties
        get_physical_device_memory_properties = nullptr;
    PFN_vkGetPhysicalDeviceProperties
        get_physical_device_properties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties
        get_physical_device_format_properties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties
        get_physical_device_queue_family_properties = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR
        get_physical_device_surface_capabilities = nullptr;
};

struct DeviceDispatch {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    float timestamp_period_ns = 0.0F;

    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;

    PFN_vkGetDeviceQueue get_device_queue = nullptr;
    PFN_vkGetDeviceQueue2 get_device_queue2 = nullptr;

    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_swapchain_images = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image = nullptr;

    PFN_vkCreateImage create_image = nullptr;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkGetImageMemoryRequirements get_image_memory_requirements = nullptr;
    PFN_vkAllocateMemory allocate_memory = nullptr;
    PFN_vkFreeMemory free_memory = nullptr;
    PFN_vkBindImageMemory bind_image_memory = nullptr;

    PFN_vkCreateCommandPool create_command_pool = nullptr;
    PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
    PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
    PFN_vkResetCommandBuffer reset_command_buffer = nullptr;
    PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
    PFN_vkEndCommandBuffer end_command_buffer = nullptr;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
    PFN_vkCmdCopyImage cmd_copy_image = nullptr;
    PFN_vkCmdBlitImage cmd_blit_image = nullptr;

    PFN_vkCreateSemaphore create_semaphore = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;
    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkDestroyFence destroy_fence = nullptr;
    PFN_vkWaitForFences wait_for_fences = nullptr;
    PFN_vkResetFences reset_fences = nullptr;

    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkQueueWaitIdle queue_wait_idle = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;

    PFN_vkSetDeviceLoaderData set_device_loader_data = nullptr;
};

struct QueueState {
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t family_index = UINT32_MAX;
    std::uint32_t queue_index = 0;
    VkDeviceQueueCreateFlags flags = 0;
    VkQueueFlags capabilities = 0;
    std::uint32_t timestamp_valid_bits = 0;
};

struct CopySlot {
    VkImage destination = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore copy_complete = VK_NULL_HANDLE;
    VkSemaphore generated_present_ready = VK_NULL_HANDLE;
    VkSemaphore source_present_ready = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool has_submission = false;
    bool used_for_present = false;
    bool generated_used_for_present = false;
    bool source_used_for_present = false;
};

struct SwapchainState {
    std::mutex mutex;

    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t generation = 0;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    VkImageUsageFlags image_usage = 0;
    std::uint32_t min_image_count = 0;
    std::uint32_t image_count = 0;

    std::vector<VkImage> images;
    std::vector<CopySlot> copy_slots;
    std::unique_ptr<ofg::vulkan::VulkanPassthroughPipeline> passthrough;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer synthetic_command_buffer = VK_NULL_HANDLE;
    VkSemaphore synthetic_acquire = VK_NULL_HANDLE;
    VkFence synthetic_fence = VK_NULL_HANDLE;
    VkQueue copy_queue = VK_NULL_HANDLE;
    std::uint32_t copy_queue_family = UINT32_MAX;
    bool synthetic_submission_pending = false;

    bool interpolate_2x_requested = false;
    bool transfer_dst_enabled = false;
    bool copy_initialized = false;
    bool copy_skip_logged = false;
    bool first_copy_logged = false;
    bool first_copy_completed_logged = false;
    bool first_passthrough_logged = false;
    bool first_sharpen_logged = false;
    bool first_history_logged = false;
    bool first_motion_logged = false;
    bool first_warp_logged = false;
    bool first_motion_validation_logged = false;
    bool first_warp_validation_logged = false;
    bool first_cadence_logged = false;
    bool first_2x_present_logged = false;
    bool first_timing_logged = false;
    bool first_present_logged = false;

    ofg::FrameCadence2xPlanner cadence_2x;
};

std::mutex g_state_mutex;
std::mutex g_log_mutex;

std::unordered_map<void*, InstanceDispatch> g_instance_dispatch;
std::unordered_map<void*, DeviceDispatch> g_device_dispatch;
std::unordered_map<VkQueue, QueueState> g_queues;
std::unordered_map<VkSwapchainKHR, std::shared_ptr<SwapchainState>>
    g_swapchains;
std::unordered_map<void*, std::vector<VkSemaphore>>
    g_retired_present_semaphores;

std::atomic<PFN_vkGetInstanceProcAddr> g_next_global_gipa{nullptr};
std::atomic<std::uint64_t> g_present_count{0};
std::atomic<std::uint64_t> g_swapchain_generation{0};

void log_message(const char* message) noexcept {
    if (message == nullptr) {
        return;
    }

    std::scoped_lock lock{g_log_mutex};

    std::fprintf(stderr, "%s\n", message);
    std::fflush(stderr);

    const char* log_path = std::getenv("OFG_LOG_FILE");
    if (log_path == nullptr || *log_path == '\0') {
        return;
    }

    if (std::FILE* file = std::fopen(log_path, "a"); file != nullptr) {
        std::fprintf(file, "%s\n", message);
        std::fclose(file);
    }
}

[[nodiscard]] const char* format_name(VkFormat format) noexcept {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_UNORM:
            return "B8G8R8A8_UNORM";
        case VK_FORMAT_B8G8R8A8_SRGB:
            return "B8G8R8A8_SRGB";
        case VK_FORMAT_R8G8B8A8_UNORM:
            return "R8G8B8A8_UNORM";
        case VK_FORMAT_R8G8B8A8_SRGB:
            return "R8G8B8A8_SRGB";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return "A2B10G10R10_UNORM_PACK32";
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            return "A2R10G10B10_UNORM_PACK32";
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return "R16G16B16A16_SFLOAT";
        default:
            return "OTHER";
    }
}

[[nodiscard]] const char* present_mode_name(
    VkPresentModeKHR present_mode) noexcept {
    switch (present_mode) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR:
            return "IMMEDIATE";
        case VK_PRESENT_MODE_MAILBOX_KHR:
            return "MAILBOX";
        case VK_PRESENT_MODE_FIFO_KHR:
            return "FIFO";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            return "FIFO_RELAXED";
        default:
            return "OTHER";
    }
}

template <typename Dispatchable>
[[nodiscard]] void* dispatch_key(Dispatchable object) noexcept {
    if (object == VK_NULL_HANDLE) {
        return nullptr;
    }

    return *reinterpret_cast<void* const*>(object);
}

[[nodiscard]] VkLayerInstanceCreateInfo* find_instance_chain_info(
    const VkInstanceCreateInfo* create_info,
    VkLayerFunction function) noexcept {
    auto* current = reinterpret_cast<const VkLayerInstanceCreateInfo*>(
        create_info->pNext);

    while (current != nullptr) {
        if (current->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
            current->function == function) {
            return const_cast<VkLayerInstanceCreateInfo*>(current);
        }

        current = reinterpret_cast<const VkLayerInstanceCreateInfo*>(
            current->pNext);
    }

    return nullptr;
}

[[nodiscard]] VkLayerDeviceCreateInfo* find_device_chain_info(
    const VkDeviceCreateInfo* create_info,
    VkLayerFunction function) noexcept {
    auto* current = reinterpret_cast<const VkLayerDeviceCreateInfo*>(
        create_info->pNext);

    while (current != nullptr) {
        if (current->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            current->function == function) {
            return const_cast<VkLayerDeviceCreateInfo*>(current);
        }

        current = reinterpret_cast<const VkLayerDeviceCreateInfo*>(
            current->pNext);
    }

    return nullptr;
}

[[nodiscard]] bool find_instance_dispatch(
    void* key,
    InstanceDispatch& out_dispatch) {
    std::scoped_lock lock{g_state_mutex};
    const auto it = g_instance_dispatch.find(key);

    if (it == g_instance_dispatch.end()) {
        return false;
    }

    out_dispatch = it->second;
    return true;
}

[[nodiscard]] bool find_device_dispatch(
    void* key,
    DeviceDispatch& out_dispatch) {
    std::scoped_lock lock{g_state_mutex};
    const auto it = g_device_dispatch.find(key);

    if (it == g_device_dispatch.end()) {
        return false;
    }

    out_dispatch = it->second;
    return true;
}

[[nodiscard]] float configured_scale_factor() noexcept {
    const char* value = std::getenv("OFG_SCALE_FACTOR");
    if (value == nullptr || *value == '\0') {
        return 1.0F;
    }

    char* end = nullptr;
    const float scale = std::strtof(value, &end);

    if (end == value ||
        end == nullptr ||
        *end != '\0' ||
        !std::isfinite(scale) ||
        scale < 0.25F ||
        scale > 2.0F) {
        return 1.0F;
    }

    return scale;
}

[[nodiscard]] float configured_sharpening_strength() noexcept {
    const char* value = std::getenv("OFG_SHARPEN_STRENGTH");
    if (value == nullptr || *value == '\0') {
        return 0.0F;
    }

    char* end = nullptr;
    const float strength = std::strtof(value, &end);

    if (end == value ||
        end == nullptr ||
        *end != '\0' ||
        !std::isfinite(strength) ||
        strength < 0.0F ||
        strength > 1.0F) {
        return 0.0F;
    }

    return strength;
}

[[nodiscard]] bool configured_motion_validation() noexcept {
    const char* value = std::getenv("OFG_MOTION_VALIDATE");

    return value != nullptr &&
           (std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "true") == 0 ||
            std::strcmp(value, "on") == 0);
}

[[nodiscard]] bool configured_2x_interpolation() noexcept {
    const char* value = std::getenv("OFG_INTERPOLATE_2X");

    return value != nullptr &&
           (std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "true") == 0 ||
            std::strcmp(value, "on") == 0);
}

[[nodiscard]] ofg::vulkan::ScaleFilter configured_scale_filter() noexcept {
    const char* value = std::getenv("OFG_SCALE_FILTER");

    if (value != nullptr &&
        std::strcmp(value, "bicubic") == 0) {
        return ofg::vulkan::ScaleFilter::Bicubic;
    }

    return ofg::vulkan::ScaleFilter::Bilinear;
}

[[nodiscard]] const char* scale_filter_name(
    ofg::vulkan::ScaleFilter filter) noexcept {
    return filter == ofg::vulkan::ScaleFilter::Bicubic
        ? "bicubic"
        : "bilinear";
}

[[nodiscard]] VkExtent2D scaled_extent(
    VkExtent2D source,
    float scale) noexcept {
    return VkExtent2D{
        std::max(
            1u,
            static_cast<std::uint32_t>(
                static_cast<double>(source.width) *
                    static_cast<double>(scale) +
                0.5)),
        std::max(
            1u,
            static_cast<std::uint32_t>(
                static_cast<double>(source.height) *
                    static_cast<double>(scale) +
                0.5)),
    };
}

[[nodiscard]] bool supports_2x_blit(
    const DeviceDispatch& dispatch,
    VkFormat destination_format) {
    InstanceDispatch instance_dispatch{};

    if (!find_instance_dispatch(
            dispatch_key(dispatch.physical_device),
            instance_dispatch) ||
        instance_dispatch.get_physical_device_format_properties == nullptr) {
        return false;
    }

    VkFormatProperties generated_properties{};
    VkFormatProperties destination_properties{};

    instance_dispatch.get_physical_device_format_properties(
        dispatch.physical_device,
        VK_FORMAT_R8G8B8A8_UNORM,
        &generated_properties);
    instance_dispatch.get_physical_device_format_properties(
        dispatch.physical_device,
        destination_format,
        &destination_properties);

    const bool generated_supported =
        (generated_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
    const bool destination_supported =
        (destination_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;

    return generated_supported && destination_supported;
}

[[nodiscard]] bool supports_scaling(
    const DeviceDispatch& dispatch,
    VkFormat source_format,
    ofg::vulkan::ScaleFilter filter,
    bool motion_validation) {
    InstanceDispatch instance_dispatch{};

    if (!find_instance_dispatch(
            dispatch_key(dispatch.physical_device),
            instance_dispatch) ||
        instance_dispatch.get_physical_device_format_properties == nullptr) {
        return false;
    }

    VkFormatProperties source_properties{};
    VkFormatProperties output_properties{};
    VkFormatProperties motion_properties{};

    instance_dispatch.get_physical_device_format_properties(
        dispatch.physical_device,
        source_format,
        &source_properties);

    instance_dispatch.get_physical_device_format_properties(
        dispatch.physical_device,
        VK_FORMAT_R8G8B8A8_UNORM,
        &output_properties);

    instance_dispatch.get_physical_device_format_properties(
        dispatch.physical_device,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        &motion_properties);

    VkFormatFeatureFlags required_source =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    if (filter == ofg::vulkan::ScaleFilter::Bilinear) {
        required_source |=
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }

    const bool source_supported =
        (source_properties.optimalTilingFeatures & required_source) ==
        required_source;

    constexpr VkFormatFeatureFlags required_output =
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

    const bool output_supported =
        (output_properties.optimalTilingFeatures & required_output) ==
        required_output;

    VkFormatFeatureFlags required_motion =
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    if (motion_validation) {
        required_motion |=
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    }

    const bool motion_supported =
        (motion_properties.optimalTilingFeatures & required_motion) ==
        required_motion;

    return source_supported &&
           output_supported &&
           motion_supported;
}

[[nodiscard]] std::uint32_t find_memory_type(
    const VkPhysicalDeviceMemoryProperties& properties,
    std::uint32_t type_bits,
    VkMemoryPropertyFlags preferred_flags) noexcept {
    for (std::uint32_t index = 0;
         index < properties.memoryTypeCount;
         ++index) {
        const bool compatible = (type_bits & (1u << index)) != 0;
        const bool preferred =
            (properties.memoryTypes[index].propertyFlags &
             preferred_flags) == preferred_flags;

        if (compatible && preferred) {
            return index;
        }
    }

    for (std::uint32_t index = 0;
         index < properties.memoryTypeCount;
         ++index) {
        if ((type_bits & (1u << index)) != 0) {
            return index;
        }
    }

    return UINT32_MAX;
}

[[nodiscard]] bool wait_for_copy_submissions(
    const DeviceDispatch& dispatch,
    SwapchainState& state) noexcept {
    std::vector<VkFence> pending_fences;
    pending_fences.reserve(state.copy_slots.size());

    for (const auto& slot : state.copy_slots) {
        if (slot.has_submission && slot.fence != VK_NULL_HANDLE) {
            pending_fences.push_back(slot.fence);
        }
    }

    if (state.synthetic_submission_pending &&
        state.synthetic_fence != VK_NULL_HANDLE) {
        pending_fences.push_back(state.synthetic_fence);
    }

    if (pending_fences.empty()) {
        return true;
    }

    if (dispatch.wait_for_fences == nullptr) {
        return false;
    }

    const VkResult result =
        dispatch.wait_for_fences(
            dispatch.device,
            static_cast<std::uint32_t>(pending_fences.size()),
            pending_fences.data(),
            VK_TRUE,
            UINT64_MAX);

    if (result != VK_SUCCESS) {
        return false;
    }

    for (auto& slot : state.copy_slots) {
        if (slot.has_submission) {
            slot.has_submission = false;
        }
    }
    state.synthetic_submission_pending = false;

    return true;
}

void destroy_copy_resources_unchecked(
    const DeviceDispatch& dispatch,
    SwapchainState& state,
    std::vector<VkSemaphore>* retired_present_semaphores = nullptr) noexcept {
    state.passthrough.reset();

    for (auto& slot : state.copy_slots) {
        if (slot.copy_complete != VK_NULL_HANDLE) {
            if (retired_present_semaphores != nullptr &&
                slot.used_for_present) {
                retired_present_semaphores->push_back(
                    slot.copy_complete);
            } else if (dispatch.destroy_semaphore != nullptr) {
                dispatch.destroy_semaphore(
                    dispatch.device,
                    slot.copy_complete,
                    nullptr);
            }

            slot.copy_complete = VK_NULL_HANDLE;
        }

        if (slot.generated_present_ready != VK_NULL_HANDLE) {
            if (retired_present_semaphores != nullptr &&
                slot.generated_used_for_present) {
                retired_present_semaphores->push_back(
                    slot.generated_present_ready);
            } else if (dispatch.destroy_semaphore != nullptr) {
                dispatch.destroy_semaphore(
                    dispatch.device,
                    slot.generated_present_ready,
                    nullptr);
            }
            slot.generated_present_ready = VK_NULL_HANDLE;
        }

        if (slot.source_present_ready != VK_NULL_HANDLE) {
            if (retired_present_semaphores != nullptr &&
                slot.source_used_for_present) {
                retired_present_semaphores->push_back(
                    slot.source_present_ready);
            } else if (dispatch.destroy_semaphore != nullptr) {
                dispatch.destroy_semaphore(
                    dispatch.device,
                    slot.source_present_ready,
                    nullptr);
            }
            slot.source_present_ready = VK_NULL_HANDLE;
        }

        if (slot.fence != VK_NULL_HANDLE &&
            dispatch.destroy_fence != nullptr) {
            dispatch.destroy_fence(
                dispatch.device,
                slot.fence,
                nullptr);
        }

        if (slot.destination != VK_NULL_HANDLE &&
            dispatch.destroy_image != nullptr) {
            dispatch.destroy_image(
                dispatch.device,
                slot.destination,
                nullptr);
        }

        if (slot.memory != VK_NULL_HANDLE &&
            dispatch.free_memory != nullptr) {
            dispatch.free_memory(
                dispatch.device,
                slot.memory,
                nullptr);
        }
    }

    state.copy_slots.clear();

    if (state.synthetic_acquire != VK_NULL_HANDLE &&
        dispatch.destroy_semaphore != nullptr) {
        dispatch.destroy_semaphore(
            dispatch.device,
            state.synthetic_acquire,
            nullptr);
    }
    state.synthetic_acquire = VK_NULL_HANDLE;

    if (state.synthetic_fence != VK_NULL_HANDLE &&
        dispatch.destroy_fence != nullptr) {
        dispatch.destroy_fence(
            dispatch.device,
            state.synthetic_fence,
            nullptr);
    }
    state.synthetic_fence = VK_NULL_HANDLE;
    state.synthetic_command_buffer = VK_NULL_HANDLE;
    state.synthetic_submission_pending = false;

    if (state.command_pool != VK_NULL_HANDLE &&
        dispatch.destroy_command_pool != nullptr) {
        dispatch.destroy_command_pool(
            dispatch.device,
            state.command_pool,
            nullptr);
    }

    state.command_pool = VK_NULL_HANDLE;
    state.copy_queue = VK_NULL_HANDLE;
    state.copy_queue_family = UINT32_MAX;
    state.copy_initialized = false;
}

void destroy_copy_resources(
    const DeviceDispatch& dispatch,
    SwapchainState& state) noexcept {
    const bool has_queue_references =
        std::any_of(
            state.copy_slots.begin(),
            state.copy_slots.end(),
            [](const CopySlot& slot) {
                return slot.has_submission ||
                       slot.used_for_present ||
                       slot.generated_used_for_present ||
                       slot.source_used_for_present;
            }) ||
        state.synthetic_submission_pending;

    if (has_queue_references &&
        state.copy_queue != VK_NULL_HANDLE &&
        dispatch.queue_wait_idle != nullptr) {
        dispatch.queue_wait_idle(state.copy_queue);
    }

    destroy_copy_resources_unchecked(dispatch, state);
}

void retire_swapchain_copy_resources(
    const DeviceDispatch& dispatch,
    SwapchainState& state,
    std::vector<VkSemaphore>& retired_present_semaphores) noexcept {
    if (!wait_for_copy_submissions(dispatch, state)) {
        log_message(
            "[OpenFrameGen] Fence-scoped copy retirement failed; "
            "falling back to vkQueueWaitIdle.");
        destroy_copy_resources(dispatch, state);
        return;
    }

    destroy_copy_resources_unchecked(
        dispatch,
        state,
        &retired_present_semaphores);
}

[[nodiscard]] bool initialize_copy_resources(
    const DeviceDispatch& dispatch,
    VkQueue queue,
    const QueueState& queue_state,
    SwapchainState& state) {
    if (state.copy_initialized) {
        if (state.copy_queue == queue &&
            state.copy_queue_family == queue_state.family_index) {
            return true;
        }

        if (!state.copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: presentation queue "
                "changed after copy resources were initialized.");
            state.copy_skip_logged = true;
        }
        return false;
    }

    if ((state.image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
        if (!state.copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: swapchain does not "
                "support VK_IMAGE_USAGE_TRANSFER_SRC_BIT.");
            state.copy_skip_logged = true;
        }
        return false;
    }

    constexpr VkQueueFlags copy_capabilities =
        VK_QUEUE_GRAPHICS_BIT |
        VK_QUEUE_COMPUTE_BIT |
        VK_QUEUE_TRANSFER_BIT;

    if ((queue_state.capabilities & copy_capabilities) == 0) {
        if (!state.copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: present queue family "
                "does not support graphics, compute, or transfer commands.");
            state.copy_skip_logged = true;
        }
        return false;
    }

    if ((queue_state.flags & VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT) != 0) {
        if (!state.copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: protected queues are "
                "not supported by the prototype.");
            state.copy_skip_logged = true;
        }
        return false;
    }

    if (state.images.empty()) {
        if (!state.copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: swapchain image handles "
                "have not been discovered yet.");
            state.copy_skip_logged = true;
        }
        return false;
    }

    if (dispatch.create_command_pool == nullptr ||
        dispatch.allocate_command_buffers == nullptr ||
        dispatch.create_image == nullptr ||
        dispatch.get_image_memory_requirements == nullptr ||
        dispatch.allocate_memory == nullptr ||
        dispatch.bind_image_memory == nullptr ||
        dispatch.create_semaphore == nullptr ||
        dispatch.create_fence == nullptr) {
        if (!state.copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: required Vulkan device "
                "functions are unavailable.");
            state.copy_skip_logged = true;
        }
        return false;
    }

    state.copy_queue = queue;
    state.copy_queue_family = queue_state.family_index;

    const float scale_factor = configured_scale_factor();
    const float sharpening_strength =
        configured_sharpening_strength();
    const bool motion_validation =
        configured_motion_validation();
    const auto scale_filter = configured_scale_filter();
    const VkExtent2D output_extent =
        scaled_extent(state.extent, scale_factor);

    const bool enable_2x_resources =
        state.interpolate_2x_requested &&
        state.transfer_dst_enabled &&
        state.present_mode == VK_PRESENT_MODE_FIFO_KHR &&
        state.images.size() >= 3 &&
        dispatch.acquire_next_image != nullptr &&
        dispatch.cmd_blit_image != nullptr &&
        supports_2x_blit(dispatch, state.format);

    const bool enable_passthrough =
        ofg::vulkan::VulkanPassthroughPipeline::build_available() &&
        (queue_state.capabilities & VK_QUEUE_COMPUTE_BIT) != 0 &&
        ofg::vulkan::VulkanPassthroughPipeline::supports_source_format(
            state.format) &&
        supports_scaling(
            dispatch,
            state.format,
            scale_filter,
            motion_validation);

    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_state.family_index,
    };

    if (dispatch.create_command_pool(
            dispatch.device,
            &pool_info,
            nullptr,
            &state.command_pool) != VK_SUCCESS) {
        log_message(
            "[OpenFrameGen] Frame copy initialization failed while creating "
            "the command pool.");
        destroy_copy_resources(dispatch, state);
        return false;
    }

    state.copy_slots.resize(state.images.size());

    VkImageUsageFlags owned_image_usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (enable_passthrough) {
        owned_image_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }

    for (std::size_t index = 0;
         index < state.copy_slots.size();
         ++index) {
        auto& slot = state.copy_slots[index];

        const VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = state.format,
            .extent = VkExtent3D{
                state.extent.width,
                state.extent.height,
                1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = owned_image_usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (dispatch.create_image(
                dispatch.device,
                &image_info,
                nullptr,
                &slot.destination) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "creating an OFG-owned image.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        VkMemoryRequirements requirements{};
        dispatch.get_image_memory_requirements(
            dispatch.device,
            slot.destination,
            &requirements);

        const std::uint32_t memory_type =
            find_memory_type(
                dispatch.memory_properties,
                requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (memory_type == UINT32_MAX) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed: no "
                "compatible GPU memory type was found.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        const VkMemoryAllocateInfo allocation_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = memory_type,
        };

        if (dispatch.allocate_memory(
                dispatch.device,
                &allocation_info,
                nullptr,
                &slot.memory) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "allocating GPU memory.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        if (dispatch.bind_image_memory(
                dispatch.device,
                slot.destination,
                slot.memory,
                0) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "binding GPU memory.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        const VkCommandBufferAllocateInfo command_buffer_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = state.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        if (dispatch.allocate_command_buffers(
                dispatch.device,
                &command_buffer_info,
                &slot.command_buffer) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "allocating a command buffer.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        if (dispatch.set_device_loader_data == nullptr ||
            dispatch.set_device_loader_data(
                dispatch.device,
                slot.command_buffer) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "initializing loader data for a command buffer.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        const VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };

        if (dispatch.create_semaphore(
                dispatch.device,
                &semaphore_info,
                nullptr,
                &slot.copy_complete) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "creating a completion semaphore.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        if (enable_2x_resources) {
            if (dispatch.create_semaphore(
                    dispatch.device,
                    &semaphore_info,
                    nullptr,
                    &slot.generated_present_ready) != VK_SUCCESS ||
                dispatch.create_semaphore(
                    dispatch.device,
                    &semaphore_info,
                    nullptr,
                    &slot.source_present_ready) != VK_SUCCESS) {
                log_message(
                    "[OpenFrameGen] 2x initialization failed while "
                    "creating presentation semaphores.");
                destroy_copy_resources(dispatch, state);
                return false;
            }
        }

        const VkFenceCreateInfo fence_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        if (dispatch.create_fence(
                dispatch.device,
                &fence_info,
                nullptr,
                &slot.fence) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] Frame copy initialization failed while "
                "creating a fence.");
            destroy_copy_resources(dispatch, state);
            return false;
        }
    }

    if (enable_2x_resources) {
        const VkCommandBufferAllocateInfo synthetic_command_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = state.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        if (dispatch.allocate_command_buffers(
                dispatch.device,
                &synthetic_command_info,
                &state.synthetic_command_buffer) != VK_SUCCESS ||
            dispatch.set_device_loader_data == nullptr ||
            dispatch.set_device_loader_data(
                dispatch.device,
                state.synthetic_command_buffer) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] 2x initialization failed while "
                "allocating the synthetic command buffer.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        const VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };

        if (dispatch.create_semaphore(
                dispatch.device,
                &semaphore_info,
                nullptr,
                &state.synthetic_acquire) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] 2x initialization failed while "
                "creating the synthetic acquire semaphore.");
            destroy_copy_resources(dispatch, state);
            return false;
        }

        const VkFenceCreateInfo synthetic_fence_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };

        if (dispatch.create_fence(
                dispatch.device,
                &synthetic_fence_info,
                nullptr,
                &state.synthetic_fence) != VK_SUCCESS) {
            log_message(
                "[OpenFrameGen] 2x initialization failed while "
                "creating the synthetic submission fence.");
            destroy_copy_resources(dispatch, state);
            return false;
        }
    }

    if (enable_passthrough) {
        std::vector<VkImage> source_images;
        source_images.reserve(state.copy_slots.size());

        for (const auto& slot : state.copy_slots) {
            source_images.push_back(slot.destination);
        }

        auto passthrough =
            std::make_unique<ofg::vulkan::VulkanPassthroughPipeline>();

        if (passthrough->initialize(
                dispatch.device,
                dispatch.get_device_proc_addr,
                dispatch.memory_properties,
                state.extent,
                output_extent,
                state.format,
                scale_filter,
                sharpening_strength,
                dispatch.timestamp_period_ns,
                queue_state.timestamp_valid_bits,
                motion_validation,
                source_images)) {
            state.passthrough = std::move(passthrough);

            char compute_message[320]{};
            std::snprintf(
                compute_message,
                sizeof(compute_message),
                "[OpenFrameGen] Vulkan %s scaler ready: "
                "%ux%u -> %ux%u, scale=%.3f, sharpen=%.3f, "
                "images=%zu, local size=8x8, timing=%s, "
                "motion-validation=%s.",
                scale_filter_name(scale_filter),
                state.extent.width,
                state.extent.height,
                output_extent.width,
                output_extent.height,
                static_cast<double>(scale_factor),
                static_cast<double>(sharpening_strength),
                state.copy_slots.size(),
                state.passthrough->timing_enabled() ? "on" : "off",
                state.passthrough->motion_validation_enabled()
                    ? "on"
                    : "off");
            log_message(compute_message);
        } else {
            log_message(
                "[OpenFrameGen] Vulkan scaler initialization failed; "
                "frame copy remains active.");
        }
    }

    state.copy_initialized = true;

    if (state.interpolate_2x_requested) {
        if (enable_2x_resources &&
            state.passthrough != nullptr &&
            state.passthrough->ready()) {
            log_message(
                "[OpenFrameGen] Vulkan 2x presentation path armed: "
                "FIFO swapchain, transfer-dst and blit supported.");
        } else {
            log_message(
                "[OpenFrameGen] Vulkan 2x presentation path unavailable; "
                "continuing with normal presentation.");
        }
    }

    char message[256]{};
    std::snprintf(
        message,
        sizeof(message),
        "[OpenFrameGen] Frame copy resources ready: %zu OFG-owned images, "
        "queue family=%u.",
        state.copy_slots.size(),
        state.copy_queue_family);
    log_message(message);

    return true;
}

[[nodiscard]] bool record_generated_present_commands(
    const DeviceDispatch& dispatch,
    SwapchainState& state,
    std::uint32_t destination_index) {
    if (state.passthrough == nullptr ||
        !state.passthrough->interpolated_frame_ready() ||
        destination_index >= state.images.size() ||
        state.synthetic_command_buffer == VK_NULL_HANDLE ||
        dispatch.reset_command_buffer == nullptr ||
        dispatch.begin_command_buffer == nullptr ||
        dispatch.end_command_buffer == nullptr ||
        dispatch.cmd_pipeline_barrier == nullptr ||
        dispatch.cmd_blit_image == nullptr) {
        return false;
    }

    const VkImage generated_image =
        state.passthrough->latest_interpolated_image();

    if (generated_image == VK_NULL_HANDLE) {
        return false;
    }

    if (dispatch.reset_command_buffer(
            state.synthetic_command_buffer,
            0) != VK_SUCCESS) {
        return false;
    }

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    if (dispatch.begin_command_buffer(
            state.synthetic_command_buffer,
            &begin_info) != VK_SUCCESS) {
        return false;
    }

    std::array<VkImageMemoryBarrier, 2> prepare{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = generated_image,
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
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = state.images[destination_index],
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        },
    };

    dispatch.cmd_pipeline_barrier(
        state.synthetic_command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        static_cast<std::uint32_t>(prepare.size()),
        prepare.data());

    const VkExtent2D generated_extent =
        state.passthrough->output_extent();

    const VkImageBlit blit{
        .srcSubresource = VkImageSubresourceLayers{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            0,
            1,
        },
        .srcOffsets = {
            VkOffset3D{0, 0, 0},
            VkOffset3D{
                static_cast<std::int32_t>(generated_extent.width),
                static_cast<std::int32_t>(generated_extent.height),
                1,
            },
        },
        .dstSubresource = VkImageSubresourceLayers{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            0,
            1,
        },
        .dstOffsets = {
            VkOffset3D{0, 0, 0},
            VkOffset3D{
                static_cast<std::int32_t>(state.extent.width),
                static_cast<std::int32_t>(state.extent.height),
                1,
            },
        },
    };

    dispatch.cmd_blit_image(
        state.synthetic_command_buffer,
        generated_image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        state.images[destination_index],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blit,
        VK_FILTER_NEAREST);

    std::array<VkImageMemoryBarrier, 2> restore{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = generated_image,
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
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = state.images[destination_index],
            .subresourceRange = VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                0,
                1,
            },
        },
    };

    dispatch.cmd_pipeline_barrier(
        state.synthetic_command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        static_cast<std::uint32_t>(restore.size()),
        restore.data());

    return dispatch.end_command_buffer(
               state.synthetic_command_buffer) == VK_SUCCESS;
}

[[nodiscard]] bool record_copy_commands(
    const DeviceDispatch& dispatch,
    const SwapchainState& state,
    std::uint32_t image_index,
    CopySlot& slot) {
    if (dispatch.reset_command_buffer == nullptr ||
        dispatch.begin_command_buffer == nullptr ||
        dispatch.end_command_buffer == nullptr ||
        dispatch.cmd_pipeline_barrier == nullptr ||
        dispatch.cmd_copy_image == nullptr) {
        return false;
    }

    if (dispatch.reset_command_buffer(
            slot.command_buffer,
            0) != VK_SUCCESS) {
        return false;
    }

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };

    if (dispatch.begin_command_buffer(
            slot.command_buffer,
            &begin_info) != VK_SUCCESS) {
        return false;
    }

    VkImageMemoryBarrier prepare_barriers[2]{};

    prepare_barriers[0] = VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = state.images[image_index],
        .subresourceRange = VkImageSubresourceRange{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        },
    };

    prepare_barriers[1] = VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = slot.destination,
        .subresourceRange = VkImageSubresourceRange{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        },
    };

    dispatch.cmd_pipeline_barrier(
        slot.command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        2,
        prepare_barriers);

    const VkImageCopy copy_region{
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
            state.extent.width,
            state.extent.height,
            1,
        },
    };

    dispatch.cmd_copy_image(
        slot.command_buffer,
        state.images[image_index],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        slot.destination,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copy_region);

    const VkImageMemoryBarrier restore_source{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = state.images[image_index],
        .subresourceRange = VkImageSubresourceRange{
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            0,
            1,
        },
    };

    dispatch.cmd_pipeline_barrier(
        slot.command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &restore_source);

    if (state.passthrough != nullptr &&
        state.passthrough->ready() &&
        !state.passthrough->record(
            slot.command_buffer,
            image_index)) {
        return false;
    }

    return dispatch.end_command_buffer(
               slot.command_buffer) == VK_SUCCESS;
}

[[nodiscard]] bool try_present_2x_pair(
    const DeviceDispatch& dispatch,
    VkQueue queue,
    const VkPresentInfoKHR* original_present,
    SwapchainState& state,
    std::uint32_t source_index,
    VkSemaphore copy_complete,
    bool cadence_plan_ready,
    VkResult& present_result) {
    if (!state.interpolate_2x_requested ||
        !state.transfer_dst_enabled ||
        state.present_mode != VK_PRESENT_MODE_FIFO_KHR ||
        !cadence_plan_ready ||
        state.passthrough == nullptr ||
        !state.passthrough->interpolated_frame_ready() ||
        state.synthetic_command_buffer == VK_NULL_HANDLE ||
        state.synthetic_acquire == VK_NULL_HANDLE ||
        state.synthetic_fence == VK_NULL_HANDLE ||
        dispatch.acquire_next_image == nullptr ||
        dispatch.queue_submit == nullptr ||
        dispatch.queue_present == nullptr ||
        dispatch.wait_for_fences == nullptr ||
        dispatch.reset_fences == nullptr ||
        source_index >= state.copy_slots.size() ||
        original_present == nullptr ||
        original_present->pSwapchains == nullptr) {
        return false;
    }

    if (state.synthetic_submission_pending) {
        const VkResult wait_result =
            dispatch.wait_for_fences(
                dispatch.device,
                1,
                &state.synthetic_fence,
                VK_TRUE,
                UINT64_MAX);

        if (wait_result != VK_SUCCESS) {
            return false;
        }

        state.synthetic_submission_pending = false;
    }

    std::uint32_t generated_index = UINT32_MAX;
    const VkSwapchainKHR swapchain =
        original_present->pSwapchains[0];

    const VkResult acquire_result =
        dispatch.acquire_next_image(
            dispatch.device,
            swapchain,
            0,
            state.synthetic_acquire,
            VK_NULL_HANDLE,
            &generated_index);

    if (acquire_result != VK_SUCCESS &&
        acquire_result != VK_SUBOPTIMAL_KHR) {
        return false;
    }

    auto release_acquired_image = [&]() noexcept {
        VkResult release_result = VK_SUCCESS;
        const VkPresentInfoKHR release_present{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &state.synthetic_acquire,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &generated_index,
            .pResults = &release_result,
        };

        dispatch.queue_present(
            queue,
            &release_present);
    };

    if (generated_index >= state.copy_slots.size() ||
        generated_index >= state.images.size() ||
        generated_index == source_index) {
        release_acquired_image();
        return false;
    }

    auto& source_slot = state.copy_slots[source_index];
    auto& generated_slot = state.copy_slots[generated_index];

    generated_slot.generated_used_for_present = false;
    generated_slot.source_used_for_present = false;
    generated_slot.used_for_present = false;

    if (generated_slot.generated_present_ready == VK_NULL_HANDLE ||
        source_slot.source_present_ready == VK_NULL_HANDLE ||
        !record_generated_present_commands(
            dispatch,
            state,
            generated_index) ||
        dispatch.reset_fences(
            dispatch.device,
            1,
            &state.synthetic_fence) != VK_SUCCESS) {
        release_acquired_image();
        return false;
    }

    const std::array<VkSemaphore, 2> wait_semaphores{
        copy_complete,
        state.synthetic_acquire,
    };
    const std::array<VkPipelineStageFlags, 2> wait_stages{
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
    };
    const std::array<VkSemaphore, 2> signal_semaphores{
        generated_slot.generated_present_ready,
        source_slot.source_present_ready,
    };

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount =
            static_cast<std::uint32_t>(wait_semaphores.size()),
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &state.synthetic_command_buffer,
        .signalSemaphoreCount =
            static_cast<std::uint32_t>(signal_semaphores.size()),
        .pSignalSemaphores = signal_semaphores.data(),
    };

    if (dispatch.queue_submit(
            queue,
            1,
            &submit_info,
            state.synthetic_fence) != VK_SUCCESS) {
        release_acquired_image();
        return false;
    }

    state.synthetic_submission_pending = true;
    source_slot.used_for_present = false;
    generated_slot.generated_used_for_present = true;
    source_slot.source_used_for_present = true;

    VkResult generated_present_result = VK_SUCCESS;
    const VkSemaphore generated_ready =
        generated_slot.generated_present_ready;
    const VkPresentInfoKHR generated_present{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &generated_ready,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &generated_index,
        .pResults = &generated_present_result,
    };

    dispatch.queue_present(
        queue,
        &generated_present);

    const VkSemaphore source_ready =
        source_slot.source_present_ready;
    VkPresentInfoKHR source_present =
        *original_present;
    source_present.waitSemaphoreCount = 1;
    source_present.pWaitSemaphores = &source_ready;

    present_result =
        dispatch.queue_present(
            queue,
            &source_present);

    if (!state.first_2x_present_logged) {
        state.first_2x_present_logged = true;

        char message[320]{};
        std::snprintf(
            message,
            sizeof(message),
            "[OpenFrameGen] First real 2x FIFO pair queued: "
            "generated image=%u before source image=%u.",
            generated_index,
            source_index);
        log_message(message);
    }

    return true;
}

VKAPI_ATTR VkResult VKAPI_CALL ofgCreateInstance(
    const VkInstanceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator,
    VkInstance* instance);

VKAPI_ATTR void VKAPI_CALL ofgDestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks* allocator);

VKAPI_ATTR VkResult VKAPI_CALL ofgCreateDevice(
    VkPhysicalDevice physical_device,
    const VkDeviceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator,
    VkDevice* device);

VKAPI_ATTR void VKAPI_CALL ofgDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* allocator);

VKAPI_ATTR void VKAPI_CALL ofgGetDeviceQueue(
    VkDevice device,
    std::uint32_t queue_family_index,
    std::uint32_t queue_index,
    VkQueue* queue);

VKAPI_ATTR void VKAPI_CALL ofgGetDeviceQueue2(
    VkDevice device,
    const VkDeviceQueueInfo2* queue_info,
    VkQueue* queue);

VKAPI_ATTR VkResult VKAPI_CALL ofgCreateSwapchainKHR(
    VkDevice device,
    const VkSwapchainCreateInfoKHR* create_info,
    const VkAllocationCallbacks* allocator,
    VkSwapchainKHR* swapchain);

VKAPI_ATTR void VKAPI_CALL ofgDestroySwapchainKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    const VkAllocationCallbacks* allocator);

VKAPI_ATTR VkResult VKAPI_CALL ofgGetSwapchainImagesKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    std::uint32_t* image_count,
    VkImage* images);

VKAPI_ATTR VkResult VKAPI_CALL ofgQueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR* present_info);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ofgGetInstanceProcAddr(
    VkInstance instance,
    const char* name);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ofgGetDeviceProcAddr(
    VkDevice device,
    const char* name);

VKAPI_ATTR VkResult VKAPI_CALL ofgCreateInstance(
    const VkInstanceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator,
    VkInstance* instance) {
    if (create_info == nullptr || instance == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* chain_info =
        find_instance_chain_info(create_info, VK_LAYER_LINK_INFO);

    if (chain_info == nullptr || chain_info->u.pLayerInfo == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const auto next_gipa =
        chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;

    if (next_gipa == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const auto next_create_instance =
        reinterpret_cast<PFN_vkCreateInstance>(
            next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));

    if (next_create_instance == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    g_next_global_gipa.store(next_gipa, std::memory_order_release);
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    const VkResult result =
        next_create_instance(create_info, allocator, instance);

    if (result != VK_SUCCESS) {
        return result;
    }

    InstanceDispatch dispatch{
        .instance = *instance,
        .get_instance_proc_addr = next_gipa,
        .destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(
            next_gipa(*instance, "vkDestroyInstance")),
        .get_physical_device_memory_properties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
                next_gipa(
                    *instance,
                    "vkGetPhysicalDeviceMemoryProperties")),
        .get_physical_device_properties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                next_gipa(
                    *instance,
                    "vkGetPhysicalDeviceProperties")),
        .get_physical_device_format_properties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
                next_gipa(
                    *instance,
                    "vkGetPhysicalDeviceFormatProperties")),
        .get_physical_device_queue_family_properties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                next_gipa(
                    *instance,
                    "vkGetPhysicalDeviceQueueFamilyProperties")),
        .get_physical_device_surface_capabilities =
            reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
                next_gipa(
                    *instance,
                    "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")),
    };

    {
        std::scoped_lock lock{g_state_mutex};
        g_instance_dispatch[dispatch_key(*instance)] = dispatch;
    }

    log_message("[OpenFrameGen] Vulkan layer attached to VkInstance.");

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL ofgDestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks* allocator) {
    InstanceDispatch dispatch{};

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_instance_dispatch.find(dispatch_key(instance));

        if (it != g_instance_dispatch.end()) {
            dispatch = it->second;
            g_instance_dispatch.erase(it);
        }
    }

    if (dispatch.destroy_instance != nullptr) {
        dispatch.destroy_instance(instance, allocator);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ofgCreateDevice(
    VkPhysicalDevice physical_device,
    const VkDeviceCreateInfo* create_info,
    const VkAllocationCallbacks* allocator,
    VkDevice* device) {
    if (create_info == nullptr || device == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    auto* chain_info =
        find_device_chain_info(create_info, VK_LAYER_LINK_INFO);

    if (chain_info == nullptr || chain_info->u.pLayerInfo == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    InstanceDispatch instance_dispatch{};
    if (!find_instance_dispatch(
            dispatch_key(physical_device),
            instance_dispatch)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const auto next_gipa =
        chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const auto next_gdpa =
        chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;

    const auto* loader_data_info =
        find_device_chain_info(
            create_info,
            VK_LOADER_DATA_CALLBACK);
    const auto set_device_loader_data =
        loader_data_info != nullptr
            ? loader_data_info->u.pfnSetDeviceLoaderData
            : nullptr;

    if (next_gipa == nullptr || next_gdpa == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const auto next_create_device =
        reinterpret_cast<PFN_vkCreateDevice>(
            next_gipa(instance_dispatch.instance, "vkCreateDevice"));

    if (next_create_device == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    const VkResult result =
        next_create_device(
            physical_device,
            create_info,
            allocator,
            device);

    if (result != VK_SUCCESS) {
        return result;
    }

    DeviceDispatch dispatch{};
    dispatch.device = *device;
    dispatch.physical_device = physical_device;
    dispatch.get_device_proc_addr = next_gdpa;
    dispatch.set_device_loader_data = set_device_loader_data;

    if (instance_dispatch.get_physical_device_memory_properties != nullptr) {
        instance_dispatch.get_physical_device_memory_properties(
            physical_device,
            &dispatch.memory_properties);
    }

    if (instance_dispatch.get_physical_device_properties != nullptr) {
        VkPhysicalDeviceProperties properties{};
        instance_dispatch.get_physical_device_properties(
            physical_device,
            &properties);
        dispatch.timestamp_period_ns =
            properties.limits.timestampPeriod;
    }

#define OFG_LOAD_DEVICE(name, field) \
    dispatch.field = reinterpret_cast<PFN_##name>( \
        next_gdpa(*device, #name))

    OFG_LOAD_DEVICE(vkDestroyDevice, destroy_device);
    OFG_LOAD_DEVICE(vkDeviceWaitIdle, device_wait_idle);
    OFG_LOAD_DEVICE(vkGetDeviceQueue, get_device_queue);
    OFG_LOAD_DEVICE(vkGetDeviceQueue2, get_device_queue2);
    OFG_LOAD_DEVICE(vkCreateSwapchainKHR, create_swapchain);
    OFG_LOAD_DEVICE(vkDestroySwapchainKHR, destroy_swapchain);
    OFG_LOAD_DEVICE(vkGetSwapchainImagesKHR, get_swapchain_images);
    OFG_LOAD_DEVICE(vkAcquireNextImageKHR, acquire_next_image);
    OFG_LOAD_DEVICE(vkCreateImage, create_image);
    OFG_LOAD_DEVICE(vkDestroyImage, destroy_image);
    OFG_LOAD_DEVICE(vkGetImageMemoryRequirements, get_image_memory_requirements);
    OFG_LOAD_DEVICE(vkAllocateMemory, allocate_memory);
    OFG_LOAD_DEVICE(vkFreeMemory, free_memory);
    OFG_LOAD_DEVICE(vkBindImageMemory, bind_image_memory);
    OFG_LOAD_DEVICE(vkCreateCommandPool, create_command_pool);
    OFG_LOAD_DEVICE(vkDestroyCommandPool, destroy_command_pool);
    OFG_LOAD_DEVICE(vkAllocateCommandBuffers, allocate_command_buffers);
    OFG_LOAD_DEVICE(vkResetCommandBuffer, reset_command_buffer);
    OFG_LOAD_DEVICE(vkBeginCommandBuffer, begin_command_buffer);
    OFG_LOAD_DEVICE(vkEndCommandBuffer, end_command_buffer);
    OFG_LOAD_DEVICE(vkCmdPipelineBarrier, cmd_pipeline_barrier);
    OFG_LOAD_DEVICE(vkCmdCopyImage, cmd_copy_image);
    OFG_LOAD_DEVICE(vkCmdBlitImage, cmd_blit_image);
    OFG_LOAD_DEVICE(vkCreateSemaphore, create_semaphore);
    OFG_LOAD_DEVICE(vkDestroySemaphore, destroy_semaphore);
    OFG_LOAD_DEVICE(vkCreateFence, create_fence);
    OFG_LOAD_DEVICE(vkDestroyFence, destroy_fence);
    OFG_LOAD_DEVICE(vkWaitForFences, wait_for_fences);
    OFG_LOAD_DEVICE(vkResetFences, reset_fences);
    OFG_LOAD_DEVICE(vkQueueSubmit, queue_submit);
    OFG_LOAD_DEVICE(vkQueueWaitIdle, queue_wait_idle);
    OFG_LOAD_DEVICE(vkQueuePresentKHR, queue_present);

#undef OFG_LOAD_DEVICE

    {
        std::scoped_lock lock{g_state_mutex};
        g_device_dispatch[dispatch_key(*device)] = dispatch;
    }

    log_message("[OpenFrameGen] Vulkan layer attached to VkDevice.");

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL ofgDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* allocator) {
    DeviceDispatch dispatch{};
    std::vector<std::shared_ptr<SwapchainState>> swapchains;
    std::vector<VkSemaphore> retired_present_semaphores;
    void* device_key = dispatch_key(device);

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_device_dispatch.find(device_key);

        if (it != g_device_dispatch.end()) {
            dispatch = it->second;
            g_device_dispatch.erase(it);
        }

        for (auto queue_it = g_queues.begin();
             queue_it != g_queues.end();) {
            if (queue_it->second.device == device) {
                queue_it = g_queues.erase(queue_it);
            } else {
                ++queue_it;
            }
        }

        for (auto swapchain_it = g_swapchains.begin();
             swapchain_it != g_swapchains.end();) {
            if (swapchain_it->second != nullptr &&
                swapchain_it->second->device == device) {
                swapchains.push_back(swapchain_it->second);
                swapchain_it = g_swapchains.erase(swapchain_it);
            } else {
                ++swapchain_it;
            }
        }

        const auto retired_it =
            g_retired_present_semaphores.find(device_key);
        if (retired_it != g_retired_present_semaphores.end()) {
            retired_present_semaphores =
                std::move(retired_it->second);
            g_retired_present_semaphores.erase(retired_it);
        }
    }

    if (dispatch.device_wait_idle != nullptr) {
        dispatch.device_wait_idle(device);
    }

    for (const auto& state : swapchains) {
        if (state == nullptr) {
            continue;
        }

        std::scoped_lock state_lock{state->mutex};
        destroy_copy_resources_unchecked(dispatch, *state);
    }

    if (dispatch.destroy_semaphore != nullptr) {
        for (VkSemaphore semaphore : retired_present_semaphores) {
            if (semaphore != VK_NULL_HANDLE) {
                dispatch.destroy_semaphore(
                    device,
                    semaphore,
                    nullptr);
            }
        }
    }

    if (dispatch.destroy_device != nullptr) {
        dispatch.destroy_device(device, allocator);
    }
}

VKAPI_ATTR void VKAPI_CALL ofgGetDeviceQueue(
    VkDevice device,
    std::uint32_t queue_family_index,
    std::uint32_t queue_index,
    VkQueue* queue) {
    DeviceDispatch dispatch{};
    if (!find_device_dispatch(dispatch_key(device), dispatch) ||
        dispatch.get_device_queue == nullptr) {
        if (queue != nullptr) {
            *queue = VK_NULL_HANDLE;
        }
        return;
    }

    dispatch.get_device_queue(
        device,
        queue_family_index,
        queue_index,
        queue);

    if (queue != nullptr && *queue != VK_NULL_HANDLE) {
        VkQueueFlags capabilities = 0;
        std::uint32_t timestamp_valid_bits = 0;

        InstanceDispatch instance_dispatch{};
        if (find_instance_dispatch(
                dispatch_key(dispatch.physical_device),
                instance_dispatch) &&
            instance_dispatch
                    .get_physical_device_queue_family_properties != nullptr) {
            std::uint32_t family_count = 0;
            instance_dispatch.get_physical_device_queue_family_properties(
                dispatch.physical_device,
                &family_count,
                nullptr);

            if (queue_family_index < family_count) {
                std::vector<VkQueueFamilyProperties> properties(
                    family_count);
                instance_dispatch
                    .get_physical_device_queue_family_properties(
                        dispatch.physical_device,
                        &family_count,
                        properties.data());

                if (queue_family_index < family_count) {
                    capabilities =
                        properties[queue_family_index].queueFlags;
                    timestamp_valid_bits =
                        properties[queue_family_index].timestampValidBits;
                }
            }
        }

        std::scoped_lock lock{g_state_mutex};
        g_queues[*queue] = QueueState{
            .device = device,
            .family_index = queue_family_index,
            .queue_index = queue_index,
            .flags = 0,
            .capabilities = capabilities,
            .timestamp_valid_bits = timestamp_valid_bits,
        };
    }
}

VKAPI_ATTR void VKAPI_CALL ofgGetDeviceQueue2(
    VkDevice device,
    const VkDeviceQueueInfo2* queue_info,
    VkQueue* queue) {
    DeviceDispatch dispatch{};
    if (!find_device_dispatch(dispatch_key(device), dispatch) ||
        dispatch.get_device_queue2 == nullptr ||
        queue_info == nullptr) {
        if (queue != nullptr) {
            *queue = VK_NULL_HANDLE;
        }
        return;
    }

    dispatch.get_device_queue2(device, queue_info, queue);

    if (queue != nullptr && *queue != VK_NULL_HANDLE) {
        VkQueueFlags capabilities = 0;
        std::uint32_t timestamp_valid_bits = 0;

        InstanceDispatch instance_dispatch{};
        if (find_instance_dispatch(
                dispatch_key(dispatch.physical_device),
                instance_dispatch) &&
            instance_dispatch
                    .get_physical_device_queue_family_properties != nullptr) {
            std::uint32_t family_count = 0;
            instance_dispatch.get_physical_device_queue_family_properties(
                dispatch.physical_device,
                &family_count,
                nullptr);

            if (queue_info->queueFamilyIndex < family_count) {
                std::vector<VkQueueFamilyProperties> properties(
                    family_count);
                instance_dispatch
                    .get_physical_device_queue_family_properties(
                        dispatch.physical_device,
                        &family_count,
                        properties.data());

                if (queue_info->queueFamilyIndex < family_count) {
                    capabilities =
                        properties[queue_info->queueFamilyIndex].queueFlags;
                    timestamp_valid_bits =
                        properties[queue_info->queueFamilyIndex]
                            .timestampValidBits;
                }
            }
        }

        std::scoped_lock lock{g_state_mutex};
        g_queues[*queue] = QueueState{
            .device = device,
            .family_index = queue_info->queueFamilyIndex,
            .queue_index = queue_info->queueIndex,
            .flags = queue_info->flags,
            .capabilities = capabilities,
            .timestamp_valid_bits = timestamp_valid_bits,
        };
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ofgCreateSwapchainKHR(
    VkDevice device,
    const VkSwapchainCreateInfoKHR* create_info,
    const VkAllocationCallbacks* allocator,
    VkSwapchainKHR* swapchain) {
    if (create_info == nullptr || swapchain == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    DeviceDispatch dispatch{};
    if (!find_device_dispatch(dispatch_key(device), dispatch) ||
        dispatch.create_swapchain == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const bool interpolate_2x_requested =
        configured_2x_interpolation();

    VkSwapchainCreateInfoKHR effective_create_info =
        *create_info;

    bool transfer_dst_enabled =
        (effective_create_info.imageUsage &
         VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0;

    if (interpolate_2x_requested &&
        effective_create_info.presentMode == VK_PRESENT_MODE_FIFO_KHR &&
        !transfer_dst_enabled) {
        InstanceDispatch instance_dispatch{};

        if (find_instance_dispatch(
                dispatch_key(dispatch.physical_device),
                instance_dispatch) &&
            instance_dispatch
                    .get_physical_device_surface_capabilities != nullptr) {
            VkSurfaceCapabilitiesKHR capabilities{};

            if (instance_dispatch
                    .get_physical_device_surface_capabilities(
                        dispatch.physical_device,
                        create_info->surface,
                        &capabilities) == VK_SUCCESS &&
                (capabilities.supportedUsageFlags &
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0) {
                effective_create_info.imageUsage |=
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                transfer_dst_enabled = true;
            }
        }
    }

    const VkResult result =
        dispatch.create_swapchain(
            device,
            &effective_create_info,
            allocator,
            swapchain);

    if (result != VK_SUCCESS) {
        return result;
    }

    auto state = std::make_shared<SwapchainState>();
    state->device = device;
    state->generation =
        g_swapchain_generation.fetch_add(
            1,
            std::memory_order_relaxed) + 1;
    state->extent = effective_create_info.imageExtent;
    state->format = effective_create_info.imageFormat;
    state->color_space = effective_create_info.imageColorSpace;
    state->present_mode = effective_create_info.presentMode;
    state->image_usage = effective_create_info.imageUsage;
    state->min_image_count = effective_create_info.minImageCount;
    state->interpolate_2x_requested = interpolate_2x_requested;
    state->transfer_dst_enabled = transfer_dst_enabled;

    {
        std::scoped_lock lock{g_state_mutex};
        g_swapchains[*swapchain] = state;
    }

    char message[512]{};
    std::snprintf(
        message,
        sizeof(message),
        "[OpenFrameGen] Swapchain #%llu created: %ux%u, "
        "format=%s(%d), present=%s(%d), minImages=%u, usage=0x%08x.",
        static_cast<unsigned long long>(state->generation),
        effective_create_info.imageExtent.width,
        effective_create_info.imageExtent.height,
        format_name(effective_create_info.imageFormat),
        static_cast<int>(effective_create_info.imageFormat),
        present_mode_name(effective_create_info.presentMode),
        static_cast<int>(effective_create_info.presentMode),
        effective_create_info.minImageCount,
        static_cast<unsigned int>(effective_create_info.imageUsage));
    log_message(message);

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL ofgDestroySwapchainKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    const VkAllocationCallbacks* allocator) {
    DeviceDispatch dispatch{};
    if (!find_device_dispatch(dispatch_key(device), dispatch) ||
        dispatch.destroy_swapchain == nullptr) {
        return;
    }

    std::shared_ptr<SwapchainState> state;

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_swapchains.find(swapchain);

        if (it != g_swapchains.end()) {
            state = it->second;
            g_swapchains.erase(it);
        }
    }

    std::vector<VkSemaphore> retired_present_semaphores;
    std::uint64_t generation = 0;
    VkExtent2D extent{};
    std::uint32_t image_count = 0;

    if (state != nullptr) {
        std::scoped_lock state_lock{state->mutex};

        retire_swapchain_copy_resources(
            dispatch,
            *state,
            retired_present_semaphores);

        generation = state->generation;
        extent = state->extent;
        image_count = state->image_count;

        dispatch.destroy_swapchain(device, swapchain, allocator);
    } else {
        dispatch.destroy_swapchain(device, swapchain, allocator);
    }

    if (!retired_present_semaphores.empty()) {
        std::scoped_lock lock{g_state_mutex};
        auto& device_semaphores =
            g_retired_present_semaphores[dispatch_key(device)];
        device_semaphores.insert(
            device_semaphores.end(),
            retired_present_semaphores.begin(),
            retired_present_semaphores.end());
    }

    if (state != nullptr) {
        char message[256]{};
        std::snprintf(
            message,
            sizeof(message),
            "[OpenFrameGen] Swapchain #%llu destroyed: %ux%u, images=%u.",
            static_cast<unsigned long long>(generation),
            extent.width,
            extent.height,
            image_count);
        log_message(message);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL ofgGetSwapchainImagesKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    std::uint32_t* image_count,
    VkImage* images) {
    DeviceDispatch dispatch{};
    if (!find_device_dispatch(dispatch_key(device), dispatch) ||
        dispatch.get_swapchain_images == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::shared_ptr<SwapchainState> state;

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_swapchains.find(swapchain);

        if (it != g_swapchains.end()) {
            state = it->second;
        }
    }

    std::unique_lock<std::mutex> state_lock;
    if (state != nullptr) {
        state_lock = std::unique_lock<std::mutex>{state->mutex};
    }

    const VkResult result =
        dispatch.get_swapchain_images(
            device,
            swapchain,
            image_count,
            images);

    if ((result != VK_SUCCESS && result != VK_INCOMPLETE) ||
        image_count == nullptr) {
        return result;
    }

    bool changed = false;
    std::uint32_t requested_minimum = 0;

    if (state != nullptr) {
        requested_minimum = state->min_image_count;

        if (state->image_count != *image_count) {
            state->image_count = *image_count;
            changed = true;
        }

        if (images != nullptr) {
            state->images.assign(
                images,
                images + *image_count);
        }
    }

    if (changed) {
        char message[256]{};
        std::snprintf(
            message,
            sizeof(message),
            "[OpenFrameGen] Swapchain images discovered: %u "
            "(requested minimum %u).",
            *image_count,
            requested_minimum);
        log_message(message);
    }

    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL ofgQueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR* present_info) {
    DeviceDispatch dispatch{};

    if (!find_device_dispatch(dispatch_key(queue), dispatch) ||
        dispatch.queue_present == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const std::uint64_t present_number =
        g_present_count.fetch_add(1, std::memory_order_relaxed) + 1;

    if (present_number == 1) {
        log_message(
            "[OpenFrameGen] First vkQueuePresentKHR intercepted.");
    }

    if (present_info == nullptr ||
        present_info->swapchainCount != 1 ||
        present_info->pSwapchains == nullptr ||
        present_info->pImageIndices == nullptr) {
        return dispatch.queue_present(queue, present_info);
    }

    QueueState queue_state{};
    bool queue_known = false;
    std::shared_ptr<SwapchainState> state;

    {
        std::scoped_lock lock{g_state_mutex};

        const auto queue_it = g_queues.find(queue);
        if (queue_it != g_queues.end()) {
            queue_state = queue_it->second;
            queue_known = true;
        }

        const auto swapchain_it =
            g_swapchains.find(present_info->pSwapchains[0]);
        if (swapchain_it != g_swapchains.end()) {
            state = swapchain_it->second;
        }
    }

    if (state == nullptr) {
        return dispatch.queue_present(queue, present_info);
    }

    std::scoped_lock state_lock{state->mutex};

    const auto present_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();

    ofg::FrameCadence2xPlan cadence_plan{};
    const bool cadence_plan_ready =
        present_time > 0 &&
        state->cadence_2x.observe_source_frame(
            static_cast<std::uint64_t>(present_time),
            cadence_plan);

    VkSemaphore copy_complete = VK_NULL_HANDLE;
    bool copy_submitted = false;

    if (!state->first_present_logged) {
        state->first_present_logged = true;

        char message[384]{};
        std::snprintf(
            message,
            sizeof(message),
            "[OpenFrameGen] First present for swapchain #%llu: "
            "%ux%u, format=%s, present=%s, images=%u.",
            static_cast<unsigned long long>(state->generation),
            state->extent.width,
            state->extent.height,
            format_name(state->format),
            present_mode_name(state->present_mode),
            state->image_count);
        log_message(message);
    }

    if (!queue_known) {
        if (!state->copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: present queue "
                "family is unknown.");
            state->copy_skip_logged = true;
        }
    } else if (queue_state.device != state->device) {
        if (!state->copy_skip_logged) {
            log_message(
                "[OpenFrameGen] Frame copy skipped: queue/device "
                "association mismatch.");
            state->copy_skip_logged = true;
        }
    } else if (
        initialize_copy_resources(
            dispatch,
            queue,
            queue_state,
            *state)) {
        const std::uint32_t image_index =
            present_info->pImageIndices[0];

        if (image_index < state->copy_slots.size() &&
            image_index < state->images.size()) {
            auto& slot = state->copy_slots[image_index];

            if (dispatch.wait_for_fences != nullptr &&
                dispatch.reset_fences != nullptr &&
                dispatch.queue_submit != nullptr) {
                const VkResult wait_result =
                    dispatch.wait_for_fences(
                        dispatch.device,
                        1,
                        &slot.fence,
                        VK_TRUE,
                        UINT64_MAX);

                if (wait_result == VK_SUCCESS &&
                    slot.has_submission &&
                    !state->first_copy_completed_logged) {
                    state->first_copy_completed_logged = true;
                    log_message(
                        "[OpenFrameGen] First GPU frame copy "
                        "completed.");
                }

                if (wait_result == VK_SUCCESS &&
                    slot.has_submission &&
                    state->passthrough != nullptr &&
                    state->passthrough->timing_enabled()) {
                    ofg::vulkan::GpuTimingSample timing{};

                    if (state->passthrough->read_timing(
                            image_index,
                            timing) &&
                        !state->first_timing_logged) {
                        state->first_timing_logged = true;

                        char timing_message[256]{};
                        std::snprintf(
                            timing_message,
                            sizeof(timing_message),
                            "[OpenFrameGen] GPU timing: "
                            "scaler=%.3f ms, sharpen=%.3f ms, "
                            "total=%.3f ms.",
                            timing.scaler_ms,
                            timing.sharpening_ms,
                            timing.total_ms);
                        log_message(timing_message);
                    }
                }

                if (wait_result == VK_SUCCESS &&
                    slot.has_submission &&
                    state->passthrough != nullptr &&
                    state->passthrough->motion_validation_enabled() &&
                    !state->first_motion_validation_logged) {
                    ofg::vulkan::MotionValidationSample validation{};

                    if (state->passthrough->read_motion_validation(
                            image_index,
                            validation)) {
                        state->first_motion_validation_logged = true;

                        constexpr float expected_x = 3.0F;
                        constexpr float expected_y = -2.0F;
                        constexpr float vector_tolerance = 0.01F;
                        constexpr float error_tolerance = 0.0001F;

                        const bool vector_matches =
                            std::fabs(
                                validation.motion_x - expected_x) <=
                                vector_tolerance &&
                            std::fabs(
                                validation.motion_y - expected_y) <=
                                vector_tolerance;
                        const bool error_matches =
                            validation.mean_error <= error_tolerance;
                        const bool sample_valid =
                            validation.valid >= 0.5F;
                        const bool passed =
                            vector_matches &&
                            error_matches &&
                            sample_valid;

                        char validation_message[320]{};
                        std::snprintf(
                            validation_message,
                            sizeof(validation_message),
                            "[OpenFrameGen] Motion validation %s: "
                            "expected=(3,-2), measured=(%.3f,%.3f), "
                            "mean-error=%.6f, valid=%.1f.",
                            passed ? "PASS" : "FAIL",
                            static_cast<double>(validation.motion_x),
                            static_cast<double>(validation.motion_y),
                            static_cast<double>(validation.mean_error),
                            static_cast<double>(validation.valid));
                        log_message(validation_message);
                    }
                }

                if (wait_result == VK_SUCCESS &&
                    slot.has_submission &&
                    state->passthrough != nullptr &&
                    state->passthrough->motion_validation_enabled() &&
                    !state->first_warp_validation_logged) {
                    ofg::vulkan::WarpValidationSample validation{};

                    if (state->passthrough->read_warp_validation(
                            image_index,
                            validation)) {
                        state->first_warp_validation_logged = true;

                        const VkExtent2D extent =
                            state->passthrough->output_extent();
                        const std::uint32_t midpoint_x =
                            extent.width / 2u;
                        const std::uint32_t midpoint_y =
                            extent.height / 2u;

                        const auto to_unorm8 =
                            [](float value) noexcept {
                                const float clamped =
                                    std::clamp(value, 0.0F, 1.0F);
                                return static_cast<std::uint8_t>(
                                    std::lround(clamped * 255.0F));
                            };

                        const float denominator_x =
                            static_cast<float>(
                                std::max(1u, extent.width - 1u));
                        const float denominator_y =
                            static_cast<float>(
                                std::max(1u, extent.height - 1u));

                        const std::array<std::uint8_t, 4>
                            expected_midpoint{
                                to_unorm8(
                                    (static_cast<float>(midpoint_x) -
                                     1.5F) /
                                    denominator_x),
                                to_unorm8(
                                    (static_cast<float>(midpoint_y) +
                                     1.0F) /
                                    denominator_y),
                                64u,
                                255u,
                            };
                        const std::array<std::uint8_t, 4>
                            expected_occlusion{
                                0u,
                                0u,
                                255u,
                                255u,
                            };

                        const auto rgba_matches =
                            [](const auto& actual,
                               const auto& expected) noexcept {
                                constexpr int tolerance = 2;

                                for (std::size_t channel = 0;
                                     channel < actual.size();
                                     ++channel) {
                                    const int difference =
                                        std::abs(
                                            static_cast<int>(
                                                actual[channel]) -
                                            static_cast<int>(
                                                expected[channel]));

                                    if (difference > tolerance) {
                                        return false;
                                    }
                                }

                                return true;
                            };

                        const bool midpoint_passed =
                            rgba_matches(
                                validation.midpoint_rgba8,
                                expected_midpoint);
                        const bool occlusion_passed =
                            rgba_matches(
                                validation.occlusion_rgba8,
                                expected_occlusion);
                        const bool passed =
                            midpoint_passed && occlusion_passed;

                        char validation_message[420]{};
                        std::snprintf(
                            validation_message,
                            sizeof(validation_message),
                            "[OpenFrameGen] Warp/occlusion validation %s: "
                            "midpoint=(%u,%u,%u,%u) "
                            "expected=(%u,%u,%u,%u), "
                            "occlusion=(%u,%u,%u,%u) "
                            "expected=(0,0,255,255).",
                            passed ? "PASS" : "FAIL",
                            static_cast<unsigned int>(
                                validation.midpoint_rgba8[0]),
                            static_cast<unsigned int>(
                                validation.midpoint_rgba8[1]),
                            static_cast<unsigned int>(
                                validation.midpoint_rgba8[2]),
                            static_cast<unsigned int>(
                                validation.midpoint_rgba8[3]),
                            static_cast<unsigned int>(
                                expected_midpoint[0]),
                            static_cast<unsigned int>(
                                expected_midpoint[1]),
                            static_cast<unsigned int>(
                                expected_midpoint[2]),
                            static_cast<unsigned int>(
                                expected_midpoint[3]),
                            static_cast<unsigned int>(
                                validation.occlusion_rgba8[0]),
                            static_cast<unsigned int>(
                                validation.occlusion_rgba8[1]),
                            static_cast<unsigned int>(
                                validation.occlusion_rgba8[2]),
                            static_cast<unsigned int>(
                                validation.occlusion_rgba8[3]));
                        log_message(validation_message);
                    }
                }

                if (wait_result == VK_SUCCESS) {
                    slot.has_submission = false;
                    slot.used_for_present = false;
                    slot.generated_used_for_present = false;
                    slot.source_used_for_present = false;
                }

                if (wait_result == VK_SUCCESS &&
                    record_copy_commands(
                        dispatch,
                        *state,
                        image_index,
                        slot) &&
                    dispatch.reset_fences(
                        dispatch.device,
                        1,
                        &slot.fence) == VK_SUCCESS) {
                    std::vector<VkPipelineStageFlags> wait_stages(
                        present_info->waitSemaphoreCount,
                        VK_PIPELINE_STAGE_TRANSFER_BIT);

                    const VkSubmitInfo submit_info{
                        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .pNext = nullptr,
                        .waitSemaphoreCount =
                            present_info->waitSemaphoreCount,
                        .pWaitSemaphores =
                            present_info->pWaitSemaphores,
                        .pWaitDstStageMask =
                            wait_stages.empty()
                                ? nullptr
                                : wait_stages.data(),
                        .commandBufferCount = 1,
                        .pCommandBuffers = &slot.command_buffer,
                        .signalSemaphoreCount = 1,
                        .pSignalSemaphores = &slot.copy_complete,
                    };

                    if (dispatch.queue_submit(
                            queue,
                            1,
                            &submit_info,
                            slot.fence) == VK_SUCCESS) {
                        copy_complete = slot.copy_complete;
                        copy_submitted = true;
                        slot.has_submission = true;

                        if (!state->first_copy_logged) {
                            state->first_copy_logged = true;

                            char message[320]{};
                            std::snprintf(
                                message,
                                sizeof(message),
                                "[OpenFrameGen] First GPU frame copy "
                                "submitted: image=%u, %ux%u, "
                                "format=%s.",
                                image_index,
                                state->extent.width,
                                state->extent.height,
                                format_name(state->format));
                            log_message(message);
                        }

                        if (state->passthrough != nullptr &&
                            state->passthrough->ready() &&
                            !state->first_passthrough_logged) {
                            state->first_passthrough_logged = true;
                            log_message(
                                "[OpenFrameGen] First Vulkan scaler "
                                "dispatch submitted.");
                        }

                        if (state->passthrough != nullptr &&
                            state->passthrough->sharpening_enabled() &&
                            !state->first_sharpen_logged) {
                            state->first_sharpen_logged = true;
                            log_message(
                                "[OpenFrameGen] First Vulkan sharpening "
                                "pass submitted.");
                        }

                        if (state->passthrough != nullptr &&
                            state->passthrough->ready()) {
                            state->passthrough->commit_frame_history();

                            if (state->passthrough->frame_history_ready() &&
                                !state->first_history_logged) {
                                state->first_history_logged = true;

                                const VkExtent2D history_extent =
                                    state->passthrough->output_extent();

                                char history_message[256]{};
                                std::snprintf(
                                    history_message,
                                    sizeof(history_message),
                                    "[OpenFrameGen] Vulkan frame history "
                                    "ready: 2 GPU-resident frames at "
                                    "%ux%u.",
                                    history_extent.width,
                                    history_extent.height);
                                log_message(history_message);
                            }

                            if (state->passthrough->motion_field_ready() &&
                                !state->first_motion_logged) {
                                state->first_motion_logged = true;

                                const VkExtent2D motion_extent =
                                    state->passthrough
                                        ->motion_field_extent();

                                char motion_message[256]{};
                                std::snprintf(
                                    motion_message,
                                    sizeof(motion_message),
                                    "[OpenFrameGen] First Vulkan motion "
                                    "estimation field ready: %ux%u "
                                    "blocks (8x8 pixels, search radius=4).",
                                    motion_extent.width,
                                    motion_extent.height);
                                log_message(motion_message);
                            }

                            if (state->passthrough->interpolated_frame_ready() &&
                                !state->first_warp_logged) {
                                state->first_warp_logged = true;

                                const VkExtent2D interpolation_extent =
                                    state->passthrough->output_extent();

                                char warp_message[256]{};
                                std::snprintf(
                                    warp_message,
                                    sizeof(warp_message),
                                    "[OpenFrameGen] First Vulkan "
                                    "occlusion-aware bidirectional midpoint "
                                    "warp ready: %ux%u at t=0.500.",
                                    interpolation_extent.width,
                                    interpolation_extent.height);
                                log_message(warp_message);
                            }

                            if (cadence_plan_ready &&
                                state->passthrough->interpolated_frame_ready() &&
                                !state->first_cadence_logged) {
                                state->first_cadence_logged = true;

                                constexpr double ns_per_ms = 1'000'000.0;
                                char cadence_message[320]{};
                                std::snprintf(
                                    cadence_message,
                                    sizeof(cadence_message),
                                    "[OpenFrameGen] 2x cadence plan ready: "
                                    "source-interval=%.3f ms, "
                                    "previous->generated=%.3f ms, "
                                    "generated->current=%.3f ms.",
                                    static_cast<double>(
                                        cadence_plan.source_interval_ns) /
                                        ns_per_ms,
                                    static_cast<double>(
                                        cadence_plan
                                            .previous_to_interpolated_ns) /
                                        ns_per_ms,
                                    static_cast<double>(
                                        cadence_plan
                                            .interpolated_to_current_ns) /
                                        ns_per_ms);
                                log_message(cadence_message);
                            }
                        }
                    } else {
                        log_message(
                            "[OpenFrameGen] Frame copy submit "
                            "failed; disabling copy resources for "
                            "this swapchain.");
                        destroy_copy_resources(dispatch, *state);
                        state->copy_skip_logged = true;
                    }
                }
            }
        }
    }

    if (!copy_submitted) {
        return dispatch.queue_present(queue, present_info);
    }

    const std::uint32_t source_index =
        present_info->pImageIndices[0];

    VkResult interpolated_present_result = VK_SUCCESS;
    if (try_present_2x_pair(
            dispatch,
            queue,
            present_info,
            *state,
            source_index,
            copy_complete,
            cadence_plan_ready,
            interpolated_present_result)) {
        return interpolated_present_result;
    }

    if (source_index < state->copy_slots.size()) {
        state->copy_slots[source_index].used_for_present = true;
    }

    VkPresentInfoKHR modified_present = *present_info;
    modified_present.waitSemaphoreCount = 1;
    modified_present.pWaitSemaphores = &copy_complete;

    return dispatch.queue_present(
        queue,
        &modified_present);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ofgGetInstanceProcAddr(
    VkInstance instance,
    const char* name) {
    if (name == nullptr) {
        return nullptr;
    }

    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetInstanceProcAddr);
    }

    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetDeviceProcAddr);
    }

    if (std::strcmp(name, "vkCreateInstance") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgCreateInstance);
    }

    if (std::strcmp(name, "vkDestroyInstance") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgDestroyInstance);
    }

    if (std::strcmp(name, "vkCreateDevice") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgCreateDevice);
    }

    if (std::strcmp(name, "vkGetDeviceQueue") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetDeviceQueue);
    }

    if (std::strcmp(name, "vkGetDeviceQueue2") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetDeviceQueue2);
    }

    if (std::strcmp(name, "vkCreateSwapchainKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgCreateSwapchainKHR);
    }

    if (std::strcmp(name, "vkDestroySwapchainKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgDestroySwapchainKHR);
    }

    if (std::strcmp(name, "vkGetSwapchainImagesKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetSwapchainImagesKHR);
    }

    if (std::strcmp(name, "vkQueuePresentKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgQueuePresentKHR);
    }

    if (instance != VK_NULL_HANDLE) {
        InstanceDispatch dispatch{};
        if (find_instance_dispatch(dispatch_key(instance), dispatch) &&
            dispatch.get_instance_proc_addr != nullptr) {
            return dispatch.get_instance_proc_addr(instance, name);
        }
    }

    const auto next_gipa =
        g_next_global_gipa.load(std::memory_order_acquire);

    return next_gipa != nullptr
        ? next_gipa(instance, name)
        : nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ofgGetDeviceProcAddr(
    VkDevice device,
    const char* name) {
    if (name == nullptr) {
        return nullptr;
    }

    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetDeviceProcAddr);
    }

    if (std::strcmp(name, "vkDestroyDevice") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgDestroyDevice);
    }

    if (std::strcmp(name, "vkGetDeviceQueue") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetDeviceQueue);
    }

    if (std::strcmp(name, "vkGetDeviceQueue2") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetDeviceQueue2);
    }

    if (std::strcmp(name, "vkCreateSwapchainKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgCreateSwapchainKHR);
    }

    if (std::strcmp(name, "vkDestroySwapchainKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgDestroySwapchainKHR);
    }

    if (std::strcmp(name, "vkGetSwapchainImagesKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgGetSwapchainImagesKHR);
    }

    if (std::strcmp(name, "vkQueuePresentKHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(
            ofgQueuePresentKHR);
    }

    if (device != VK_NULL_HANDLE) {
        DeviceDispatch dispatch{};
        if (find_device_dispatch(dispatch_key(device), dispatch) &&
            dispatch.get_device_proc_addr != nullptr) {
            return dispatch.get_device_proc_addr(device, name);
        }
    }

    return nullptr;
}

} // namespace

#if defined(_WIN32)
#define OFG_LAYER_EXPORT extern "C" __declspec(dllexport)
#else
#define OFG_LAYER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

OFG_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface* version_struct) {
    if (version_struct == nullptr ||
        version_struct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT ||
        version_struct->loaderLayerInterfaceVersion < 2) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    version_struct->loaderLayerInterfaceVersion =
        std::min(version_struct->loaderLayerInterfaceVersion, 2u);
    version_struct->pfnGetInstanceProcAddr = ofgGetInstanceProcAddr;
    version_struct->pfnGetDeviceProcAddr = ofgGetDeviceProcAddr;
    version_struct->pfnGetPhysicalDeviceProcAddr = nullptr;

    return VK_SUCCESS;
}
