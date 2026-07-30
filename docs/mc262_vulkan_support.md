---
title: Minecraft 26.2 & Vulkan Support
nav_order: 8
---

# Minecraft 26.2 & Vulkan Support

Starting with **CraftGround 2.7.8**, CraftGround supports **Minecraft 26.2** in addition to the existing 1.21.0 environment. The two versions are built and shipped as separate environments (`minecraft/mc121/` and `minecraft/mc262/` in the source tree), so existing 1.21.0 experiments are unaffected.

## OpenGL and Vulkan rendering backends

Minecraft 26.2 can render using either **OpenGL** or **Vulkan**. Earlier capture code paths were tied directly to OpenGL (`glReadPixels`), which meant no frames could be captured when running under Vulkan. CraftGround 2.7.8 adds a backend-neutral frame capture path for mc262 that works correctly on both backends, supporting RAW, PNG, depth, and stereo capture modes.

## Zero-copy GPU interop (Vulkan)

When running Minecraft 26.2 with the Vulkan backend, CraftGround can now hand frames to your ML framework via **zero-copy** GPU-to-GPU transfer instead of a CPU round-trip, on two platforms:

- **Vulkan + CUDA** (Linux/Windows) — via `VK_KHR_external_memory_fd` / `VK_KHR_external_memory_win32`.
- **Vulkan + Metal** (macOS) — via `VK_EXT_metal_objects`.

This significantly reduces per-step overhead for GPU-based training pipelines compared to reading pixels back through the CPU.

## Further reading

For implementation-level details (decompiled engine internals, frame/fence ordering, capture pipeline design), see the internal engineering notes in [`26_2_vulkan_capture.md`](26_2_vulkan_capture.md) and [`26_2_MigrationPlan.md`](26_2_MigrationPlan.md).
