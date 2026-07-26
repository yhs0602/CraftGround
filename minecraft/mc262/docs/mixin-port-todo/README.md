# Deferred mixin ports

These files are copies of the corresponding `minecraft/mc121` mixins with **only the package
rename applied where trivial** — they are not compiled (not under `src/`) and not registered in
`minecraftenv.mixins.json` / `minecraftenv.client.mixins.json`, because their mc121 target
APIs don't have a mechanical 1:1 equivalent in Minecraft 26.2's Mojmap mappings:

- `ClientRenderInvoker`: no `Minecraft.render(boolean)` method was found under that
  name/signature - needs re-investigation against the 26.2 render loop entry point.
- `WindowOffScreenMixin`: `WindowProvider.createWindow` doesn't exist; window creation moved
  into `com.mojang.blaze3d.platform.Window`'s own constructor/`createWindow` (now private).
- `GameRendererDepthCaptureMixin`, `RenderLayerTrianglesMixin`: both depend on
  `Tessellator`/`VertexConsumerProvider`/`RenderPhase`/`RenderLayer.MultiPhaseParameters`, none
  of which exist by that name in the decompiled 26.2 sources
  (`com.mojang.blaze3d.vertex.VertexConsumer` exists, but the
  `Tesselator`/`MultiBufferSource`/render-phase-builder trio doesn't) - this is the
  RenderPipeline/Blaze3D rewrite `docs/26_2_MigrationPlan.md` section 6 warned about, and
  needs an actual redesign against the new rendering API, not a mechanical port.
  `GameRendererDepthCaptureMixin` specifically backs depth capture, which is still deferred/
  unported for mc262 - `GameRendererDepthCaptureMixinGetterInterface` in `src/client/` has a
  real consumer in `MinecraftEnv.kt` but no mixin implementation yet.

Everything else that originally lived in this directory (`EntityCollisionDetectorMixin`,
`AbstractBlockCollisionDetectorMixin`, `ServerPlayNetworkHandlerDisableSpamChecker`,
`GammaMixin`, `ChatVisibleMessageAccessor`, `RenderMixin`, `InputUtilMixin`,
`RenderTickCounterAccessor`, `TickSpeedMixin`, `ClientWorldMixin`,
`WorldRendererCallEntityRenderMixin`) has since actually been ported - either under the same
name against the real 26.2 API, or redesigned/renamed (`InputUtilMixin` -> `InputConstantsMixin`;
`RenderTickCounterAccessor`/`TickSpeedMixin` -> `DeltaTrackerTimerAccessor`/`ClientTickPinMixin`;
`ClientWorldMixin`/`WorldRendererCallEntityRenderMixin` -> `LevelExtractorEntityListenerMixin`) -
and removed from this directory. Check `src/main/resources/minecraftenv.mixins.json` and
`src/client/resources/minecraftenv.client.mixins.json` for the current registered list; this
README only tracks the genuinely-deferred remainder above, not the full original mc121 mixin
set.

See `docs/26_2_MigrationPlan.md` for the broader migration status. To continue this work:
run `./gradlew genSources` (or `genSourcesWithVineflower`) in `minecraft/mc262` to get local
decompiled 26.2 sources under `.gradle/loom-cache/minecraftMaven/.../*-sources.jar`, unzip them,
and grep for the actual current API before touching these files.
