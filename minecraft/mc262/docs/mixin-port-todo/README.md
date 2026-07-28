# Deferred mixin ports

**Nothing is deferred anymore.** This directory used to hold copies of mc121 mixins whose mc121
target APIs had no mechanical 1:1 equivalent in Minecraft 26.2's Mojmap mappings. All of them have
now been resolved, either by porting or by a deliberate decision not to port:

| mixin | outcome |
|---|---|
| `GameRendererDepthCaptureMixin` | **Ported.** Redesigned against the 26.2 API and now lives in `src/client/java/.../mixin/`, registered in `minecraftenv.client.mixins.json`. It is texture-based (`RenderTarget.getDepthTexture()` -> `GlTexture.glId()`) rather than FBO-based, and accounts for 26.2's reverse-Z depth range. It must inject right after `GameRenderer.renderLevel`, because `GameRenderer.render` clears the depth texture before the GUI pass - by the time the color capture runs at `CommandEncoder.submit()`, world depth no longer exists. See `src/main/cpp/depth_capture.cpp`. |
| `RenderLayerTrianglesMixin` | **Not ported, deliberately.** Its only consumer is `customentity/ModelCache.kt`'s `skull_and_roses_triangles` texture, i.e. the REALISTIC_HUMAN custom entity, which is out of scope per `docs/26_2_phase2_plan.md` §4 (D4). It would be dead code in mc262. |
| `ClientRenderInvoker` | **Deleted (W10).** Its mc121 call sites were all commented out, and the W1 present-hook design does not need it; the stereo capture path drives `GameRenderer.update/extract/render` directly instead. |
| `WindowOffScreenMixin` | **Deleted (W10).** mc121's own copy was entirely commented-out dead code, and 26.2 has no `WindowProvider.createWindow` to target - window creation moved into `com.mojang.blaze3d.platform.Window`'s own (private) constructor path. |

Everything else that originally lived here (`EntityCollisionDetectorMixin`,
`AbstractBlockCollisionDetectorMixin`, `ServerPlayNetworkHandlerDisableSpamChecker`, `GammaMixin`,
`ChatVisibleMessageAccessor`, `RenderMixin`, `InputUtilMixin`, `RenderTickCounterAccessor`,
`TickSpeedMixin`, `ClientWorldMixin`, `WorldRendererCallEntityRenderMixin`) was ported earlier -
either under the same name against the real 26.2 API, or redesigned/renamed (`InputUtilMixin` ->
`InputConstantsMixin`; `RenderTickCounterAccessor`/`TickSpeedMixin` ->
`DeltaTrackerTimerAccessor`/`ClientTickPinMixin`; `ClientWorldMixin`/
`WorldRendererCallEntityRenderMixin` -> `LevelExtractorEntityListenerMixin`).

`src/main/resources/minecraftenv.mixins.json` and `src/client/resources/minecraftenv.client.mixins.json`
are the authoritative list of what is registered. See `docs/26_2_MigrationPlan.md` and
`docs/26_2_phase2_plan.md` for the broader migration status.

This file is kept (rather than deleted with the directory) so the resolution of each deferred item
stays discoverable; it can go once `docs/26_2_phase2_plan.md` is retired.
