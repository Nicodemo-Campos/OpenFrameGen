// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

struct InstanceDispatch {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
};

struct DeviceDispatch {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_swapchain_images = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;
};

struct SwapchainState {
    VkDevice device = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    VkImageUsageFlags image_usage = 0;
    std::uint32_t min_image_count = 0;
    std::uint32_t image_count = 0;
    bool first_present_logged = false;
};

std::mutex g_state_mutex;
std::mutex g_log_mutex;

std::unordered_map<void*, InstanceDispatch> g_instance_dispatch;
std::unordered_map<void*, DeviceDispatch> g_device_dispatch;
std::unordered_map<VkSwapchainKHR, SwapchainState> g_swapchains;

std::atomic<PFN_vkGetInstanceProcAddr> g_next_global_gipa{nullptr};
std::atomic<std::uint64_t> g_present_count{0};

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

    DeviceDispatch dispatch{
        .device = *device,
        .get_device_proc_addr = next_gdpa,
        .destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(
            next_gdpa(*device, "vkDestroyDevice")),
        .create_swapchain = reinterpret_cast<PFN_vkCreateSwapchainKHR>(
            next_gdpa(*device, "vkCreateSwapchainKHR")),
        .destroy_swapchain = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
            next_gdpa(*device, "vkDestroySwapchainKHR")),
        .get_swapchain_images =
            reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(
                next_gdpa(*device, "vkGetSwapchainImagesKHR")),
        .queue_present = reinterpret_cast<PFN_vkQueuePresentKHR>(
            next_gdpa(*device, "vkQueuePresentKHR")),
    };

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

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_device_dispatch.find(dispatch_key(device));

        if (it != g_device_dispatch.end()) {
            dispatch = it->second;
            g_device_dispatch.erase(it);
        }

        for (auto it_swapchain = g_swapchains.begin();
             it_swapchain != g_swapchains.end();) {
            if (it_swapchain->second.device == device) {
                it_swapchain = g_swapchains.erase(it_swapchain);
            } else {
                ++it_swapchain;
            }
        }
    }

    if (dispatch.destroy_device != nullptr) {
        dispatch.destroy_device(device, allocator);
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

    const VkResult result =
        dispatch.create_swapchain(
            device,
            create_info,
            allocator,
            swapchain);

    if (result != VK_SUCCESS) {
        return result;
    }

    const SwapchainState state{
        .device = device,
        .extent = create_info->imageExtent,
        .format = create_info->imageFormat,
        .color_space = create_info->imageColorSpace,
        .present_mode = create_info->presentMode,
        .image_usage = create_info->imageUsage,
        .min_image_count = create_info->minImageCount,
        .image_count = 0,
        .first_present_logged = false,
    };

    {
        std::scoped_lock lock{g_state_mutex};
        g_swapchains[*swapchain] = state;
    }

    char message[512]{};
    std::snprintf(
        message,
        sizeof(message),
        "[OpenFrameGen] Swapchain created: %ux%u, format=%s(%d), "
        "present=%s(%d), minImages=%u, usage=0x%08x.",
        state.extent.width,
        state.extent.height,
        format_name(state.format),
        static_cast<int>(state.format),
        present_mode_name(state.present_mode),
        static_cast<int>(state.present_mode),
        state.min_image_count,
        static_cast<unsigned int>(state.image_usage));
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

    SwapchainState state{};
    bool tracked = false;

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_swapchains.find(swapchain);

        if (it != g_swapchains.end()) {
            state = it->second;
            tracked = true;
            g_swapchains.erase(it);
        }
    }

    dispatch.destroy_swapchain(device, swapchain, allocator);

    if (tracked) {
        char message[256]{};
        std::snprintf(
            message,
            sizeof(message),
            "[OpenFrameGen] Swapchain destroyed: %ux%u, images=%u.",
            state.extent.width,
            state.extent.height,
            state.image_count);
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

    const VkResult result =
        dispatch.get_swapchain_images(
            device,
            swapchain,
            image_count,
            images);

    if (result != VK_SUCCESS || image_count == nullptr) {
        return result;
    }

    bool changed = false;
    SwapchainState state{};

    {
        std::scoped_lock lock{g_state_mutex};
        const auto it = g_swapchains.find(swapchain);

        if (it != g_swapchains.end() &&
            it->second.image_count != *image_count) {
            it->second.image_count = *image_count;
            state = it->second;
            changed = true;
        }
    }

    if (changed) {
        char message[256]{};
        std::snprintf(
            message,
            sizeof(message),
            "[OpenFrameGen] Swapchain images discovered: %u "
            "(requested minimum %u).",
            state.image_count,
            state.min_image_count);
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

    if (present_info != nullptr &&
        present_info->pSwapchains != nullptr) {
        for (std::uint32_t index = 0;
             index < present_info->swapchainCount;
             ++index) {
            SwapchainState state{};
            bool should_log = false;

            {
                std::scoped_lock lock{g_state_mutex};
                const auto it =
                    g_swapchains.find(present_info->pSwapchains[index]);

                if (it != g_swapchains.end() &&
                    !it->second.first_present_logged) {
                    it->second.first_present_logged = true;
                    state = it->second;
                    should_log = true;
                }
            }

            if (should_log) {
                char message[384]{};
                std::snprintf(
                    message,
                    sizeof(message),
                    "[OpenFrameGen] First present for tracked swapchain: "
                    "%ux%u, format=%s, present=%s, images=%u.",
                    state.extent.width,
                    state.extent.height,
                    format_name(state.format),
                    present_mode_name(state.present_mode),
                    state.image_count);
                log_message(message);
            }
        }
    }

    return dispatch.queue_present(queue, present_info);
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
