# OpenFrameGen roadmap

This roadmap describes direction, not release promises.

## Phase 0 — Bootstrap

- [x] Public repository and MPL-2.0 license
- [x] Cross-platform CMake project
- [x] Minimal OFG core
- [x] Minimal CLI frontend
- [x] Portable smoke test
- [x] CI verified on Linux and Windows
- [ ] Define logging and error model

## Phase 1 — Vulkan presentation layer

- [x] Load as a Vulkan layer
- [x] Intercept instance/device creation safely
- [x] Track swapchain creation and destruction
- [x] Observe `vkQueuePresentKHR`
- [x] Copy a presented image into OFG-owned GPU resources
- [ ] Preserve synchronization correctness
- [x] Add validation-layer test instructions

## Phase 2 — First image-processing pipeline

- [x] Compute backend abstraction
- [ ] Pass-through compute pipeline
- [ ] Bilinear/bicubic scaler
- [ ] Sharpening pass
- [ ] Timing and GPU-cost instrumentation

## Phase 3 — Frame interpolation

- [ ] Frame history
- [ ] Motion estimation prototype
- [ ] Bidirectional warping
- [ ] Occlusion handling
- [ ] 2x interpolation
- [ ] Frame pacing and latency metrics

## Phase 4 — User experience

- [ ] Persistent configuration
- [ ] Game/application profiles
- [ ] GUI frontend
- [ ] Overlay
- [ ] Linux packaging
- [ ] Windows packaging

## Later

- [ ] DirectX/DXGI backend
- [ ] Hardware optical-flow backends where available
- [ ] Additional upscalers
- [ ] Plugin/module SDK
- [ ] HDR and VRR work
- [ ] macOS/Metal investigation
