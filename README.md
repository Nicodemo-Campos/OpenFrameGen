# OpenFrameGen

```text
   ___                   ______                          ______
  / _ \ _ __   ___ _ __ |  ___| __ __ _ _ __ ___   ___/ ___| ___ _ __
 | | | | '_ \ / _ \ '_ \| |_ | '__/ _` | '_ ` _ \ / _ \ |  _ / _ \ '_ \
 | |_| | |_) |  __/ | | |  _|| | | (_| | | | | | |  __/ |_| |  __/ | | |
  \___/| .__/ \___|_| |_|_|  |_|  \__,_|_| |_| |_|\___|\____|\___|_| |_|
       |_|

                 Open Frame Generation
```

**OpenFrameGen (OFG)** is a free and open-source cross-platform framework for frame generation, interpolation, upscaling and presentation.

> Frame generation, without the platform lock-in.

## Status

OpenFrameGen is in **very early development**. The current codebase is only the portable project bootstrap: a small core library, a CLI frontend and cross-platform tests.

Frame generation, Vulkan interception, upscaling and the GUI are **not implemented yet**.

## Goals

- Linux and Windows as the first-class development targets.
- macOS as a future platform without forcing a core redesign.
- CLI and GUI frontends backed by the same core.
- Vulkan-first graphics architecture with additional backends later.
- Modular frame-generation, optical-flow and upscaling algorithms.
- Avoid dependence on a particular Linux display server.
- Keep the project free and open source.

## Planned architecture

```text
                         OpenFrameGen
                              |
                       +------+------+
                       |   OFG Core  |
                       +------+------+
                              |
             +----------------+----------------+
             |                |                |
            CLI              GUI            Overlay
             |                |                |
             +----------------+----------------+
                              |
                     Compute / Backend API
                              |
                 +------------+------------+
                 |            |            |
              Vulkan        D3D12        Metal
                 |            |            |
              Linux        Windows       macOS
```

The algorithms are intended to sit above the graphics API so that interpolation logic can be shared between platforms.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design direction and [docs/ROADMAP.md](docs/ROADMAP.md) for the current plan.

## Build

Requirements for the bootstrap:

- CMake 3.20+
- A C++20 compiler

### Linux / macOS

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/bin/ofg info
```

### Windows

From a Developer PowerShell or another environment with CMake and a C++ compiler:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\bin\Release\ofg.exe info
```

Depending on the CMake generator, the executable may instead be located directly under `build/bin`.

## CLI

The bootstrap CLI currently exposes only diagnostic commands:

```text
ofg info
ofg version
ofg help
```

Later the CLI and GUI will control the same profiles and runtime.

## Contributing

OpenFrameGen is being built in small, reviewable steps. Architecture and APIs will change substantially while the project is young.

Bug reports, design discussion and platform testing will become increasingly useful as the first Vulkan backend lands.

## License

OpenFrameGen is licensed under the [Mozilla Public License 2.0](LICENSE).
