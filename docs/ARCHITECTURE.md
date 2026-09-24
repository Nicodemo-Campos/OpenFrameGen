# OpenFrameGen architecture

OpenFrameGen is designed as a cross-platform framework rather than a single monolithic application.

## Design principles

1. **One core, multiple frontends.** CLI, GUI and the future in-game overlay must use the same core APIs.
2. **Platform-independent algorithms.** Frame interpolation, optical flow, upscaling and frame pacing should not depend directly on Win32, Wayland, X11 or Metal.
3. **Replaceable GPU backends.** Vulkan is the first backend. Direct3D 12 is planned for Windows, and Metal is reserved for future macOS work.
4. **Desktop capture is a fallback.** On Linux, OpenFrameGen should process frames as close to the application's rendering pipeline as possible.
5. **Modules over hard-wiring.** Backends and algorithms should become swappable modules as the API stabilizes.

## Planned layers

```text
CLI / GUI / Overlay
        |
     OFG Core
        |
+-------+--------+----------------+
|                |                |
Frame pipeline   Profiles         Module API
|                                 |
+------------+--------------------+
             |
       Compute abstraction
             |
      +------+------+ 
      |             |
   Vulkan        D3D12          Metal
      |             |             |
    Linux        Windows        macOS
```

## Initial repository layout

```text
OpenFrameGen/
├── core/               Core APIs and platform-independent state
├── frontends/
│   └── cli/            Command-line frontend
├── docs/               Design and development documentation
├── tests/              Portable smoke/unit tests
├── LICENSE
└── CMakeLists.txt
```

Directories for Vulkan, capture, algorithms, shaders and additional frontends will be added when their first implementation lands rather than being committed empty.

## Near-term backend plan

### Linux

Primary path:

```text
Game -> Vulkan / Proton translation layer -> OFG Vulkan layer -> compositor -> display
```

Wayland and X11 are integrations, not dependencies of the core.

### Windows

The Vulkan path can be shared with Linux for Vulkan applications. Native DirectX support will later use a dedicated DXGI / Direct3D backend.

### macOS

macOS is a future target. The core must remain portable enough that a Metal backend and native app frontend can be introduced without redesigning the algorithms.
