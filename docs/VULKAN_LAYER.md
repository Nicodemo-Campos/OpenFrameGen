# Vulkan layer development

OpenFrameGen's first graphics integration is an explicit Vulkan layer.

The current milestone is intentionally small: negotiate with the Vulkan loader, pass instance/device creation down the layer chain, and observe `vkQueuePresentKHR` without modifying frames.

## Requirements

- CMake 3.20+
- C++20 compiler
- Vulkan headers containing `vulkan/vk_layer.h`

The first layer does **not** link directly against the Vulkan loader. It receives the next dispatch functions from the loader/layer chain.

### Windows with MSYS2 UCRT64

The following packages are sufficient for the current layer:

```text
mingw-w64-ucrt-x86_64-vulkan-headers
mingw-w64-ucrt-x86_64-vulkan-loader
```

A `VULKAN_SDK` environment variable is not required when the compiler can already find the MSYS2 headers.

## Build

```powershell
cmake -S . -B build -DOFG_BUILD_VULKAN_LAYER=ON
cmake --build build
```

If Vulkan headers are found, CMake prints their include directory and writes the development layer into:

```text
build/ofg-layer/
├── VkLayer_OpenFrameGen.json
└── openframegen_vulkan_layer.dll   # Windows
```

Linux produces the equivalent shared object.

## Discover the development layer on Windows

From the repository root:

```powershell
$env:VK_LAYER_PATH = (Resolve-Path .\build\ofg-layer).Path
vulkaninfo --summary | Select-String "OPENFRAMEGEN"
```

The layer should appear as:

```text
VK_LAYER_OPENFRAMEGEN_framegen
```

## Activate it for a process

```powershell
$env:VK_LAYER_PATH = (Resolve-Path .\build\ofg-layer).Path
$env:VK_INSTANCE_LAYERS = "VK_LAYER_OPENFRAMEGEN_framegen"
vulkaninfo --summary
```

For GUI applications on Windows, set `OFG_LOG_FILE` because their stderr output may not be visible in PowerShell:

```powershell
$env:OFG_LOG_FILE = (Join-Path (Get-Location) "ofg-layer.log")
Remove-Item $env:OFG_LOG_FILE -ErrorAction SilentlyContinue
vkcube
Start-Sleep -Seconds 2
Get-Content $env:OFG_LOG_FILE
```

During instance/device creation, the development layer writes diagnostic messages to stderr.

A program that actually presents through Vulkan should eventually produce:

```text
[OpenFrameGen] Swapchain created: 1280x720, format=B8G8R8A8_UNORM(...), present=FIFO(...), minImages=...
[OpenFrameGen] Swapchain images discovered: ...
[OpenFrameGen] First vkQueuePresentKHR intercepted.
[OpenFrameGen] First present for tracked swapchain: 1280x720, format=..., present=..., images=...
```

The exact resolution, image count, format and present mode depend on the application and driver.

`vulkaninfo` does not necessarily present frames, so not seeing the presentation message there is expected.

After testing:

```powershell
Remove-Item Env:VK_INSTANCE_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:VK_LAYER_PATH -ErrorAction SilentlyContinue
Remove-Item Env:OFG_LOG_FILE -ErrorAction SilentlyContinue
```

## Scope of this milestone

This code does not yet:

- copy swapchain images;
- create OFG-owned GPU images;
- modify synchronization;
- generate frames;
- upscale frames;
- install itself as an implicit layer.

Those features will be introduced only after pass-through presentation and swapchain tracking are stable.

## Swapchain tracking milestone

The layer now observes:

- `vkCreateSwapchainKHR`;
- `vkGetSwapchainImagesKHR`;
- `vkDestroySwapchainKHR`;
- the first `vkQueuePresentKHR` that references each tracked swapchain.

This records the swapchain extent, image format, present mode, requested minimum image count, actual discovered image count and image-usage flags without modifying the application's images or synchronization.


## Frame copy prototype

The frame-copy milestone inserts one queue submission before presentation when all of these conditions are true:

- exactly one swapchain is being presented;
- the swapchain exposes `VK_IMAGE_USAGE_TRANSFER_SRC_BIT`;
- the present queue has been observed through `vkGetDeviceQueue` or `vkGetDeviceQueue2`;
- swapchain image handles have already been discovered.

For each swapchain image, OpenFrameGen creates an OFG-owned `VkImage`, GPU memory, a command buffer, a binary semaphore and a fence. The copy submission waits on the application's original present semaphores, copies the selected swapchain image, signals OFG's semaphore, and presentation waits on that semaphore.

The destination image remains internal to OFG and is not displayed or read back yet.

### Expected test log

With `vkcube` and `OFG_LOG_FILE` enabled, a successful prototype should include:

```text
[OpenFrameGen] Frame copy resources ready: 3 OFG-owned images, queue family=...
[OpenFrameGen] First GPU frame copy submitted: image=..., 500x500, format=B8G8R8A8_UNORM.
```

If the application keeps rendering normally after those messages, the first GPU-side frame copy path is working.
