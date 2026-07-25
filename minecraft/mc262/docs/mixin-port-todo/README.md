# Deferred mixin ports

These files are copies of the corresponding `minecraft/mc121` mixins with **only the package
rename applied where trivial** — they are not compiled (not under `src/`) and not registered in
`minecraftenv.mixins.json` / `minecraftenv.client.mixins.json`, because their mc121 target
APIs don't have a mechanical 1:1 equivalent in Minecraft 26.2's Mojmap mappings:

- `AbstractBlockCollisionDetectorMixin`, `EntityCollisionDetectorMixin`: mc121's
  `onEntityCollision`/`onPlayerCollision` hooks aren't present under those names on
  `BlockBehaviour`/`Entity` in 26.2 - the collision callback surface was restructured.
- `ServerPlayNetworkHandlerDisableSpamChecker`: `checkForSpam` no longer exists;
  `ServerGamePacketListenerImpl` now rate-limits chat/commands via `TickThrottler` fields
  (`chatSpamThrottler`, `commandSpamThrottler`, `detectCommandRateSpam()`).
- `ChatVisibleMessageAccessor`: `ChatHudLine`/`visibleMessages` don't exist; `ChatComponent`
  (formerly `ChatHud`) now stores `List<GuiMessage> allMessages` instead.
- `ClientWorldMixin`: the `@Redirect` target `ClientWorld.getEntities()` isn't called from
  `WorldRenderer.render` anymore; `LevelRenderer` (formerly `WorldRenderer`) calls
  `ClientLevel.entitiesForRendering()`, and `ClientLevel.getEntities()` returns a
  `LevelEntityGetter`, not `Iterable<Entity>`.
- `ClientRenderInvoker`: no `Minecraft.render(boolean)` method was found under that
  name/signature - needs re-investigation against the 26.2 render loop entry point.
- `GammaMixin`: `SimpleOption` (now `OptionInstance`) renamed its `text` field to `caption`,
  and no longer exposes `getCodec()`/`setValue()` with matching signatures (`codec()` exists,
  `setValue` wasn't confirmed) - needs re-verification, not a blind rename.
- `InputUtilMixin`: `InputUtil` (now `InputConstants`) renamed every overwritten static method
  (`isKeyPressed` -> `isKeyDown`, `setMouseCallbacks` -> `setupMouseCallbacks`,
  `setKeyboardCallbacks` -> `setupKeyboardCallbacks`, `setCursorParameters` ->
  `grabOrReleaseMouse`) and changed their parameter shapes (they now take a `Window`
  object instead of a raw `long` handle in some cases) - needs a rewrite, not a rename.
- `WindowOffScreenMixin`: `WindowProvider.createWindow` doesn't exist; window creation moved
  into `com.mojang.blaze3d.platform.Window`'s own constructor/`createWindow` (now private).
- `RenderTickCounterAccessor`, `TickSpeedMixin`: `RenderTickCounter.Dynamic` was replaced by
  the `DeltaTracker` interface (default impl `DeltaTracker.DefaultValue`, mutable impl
  `DeltaTracker.Timer`), with entirely different fields
  (`deltaTicks`/`deltaTickResidual`/`realtimeDeltaTicks`/`lastMs` instead of
  `prevTimeMillis`/`lastFrameDuration`/`tickDelta`) and no `beginRenderTick` method found.
- `RenderMixin`, `GameRendererDepthCaptureMixin`, `RenderLayerTrianglesMixin`,
  `WorldRendererCallEntityRenderMixin`: all depend on `Tessellator`/`VertexConsumerProvider`/
  `RenderPhase`/`RenderLayer.MultiPhaseParameters`, none of which exist by that name in the
  decompiled 26.2 sources (`com.mojang.blaze3d.vertex.VertexConsumer` exists, but the
  `Tesselator`/`MultiBufferSource`/render-phase-builder trio doesn't) - this is the
  RenderPipeline/Blaze3D rewrite `docs/26_2_MigrationPlan.md` section 6 warned about, and
  needs an actual redesign against the new rendering API, not a mechanical port.

See `docs/26_2_MigrationPlan.md` for the broader migration status. To continue this work:
run `./gradlew genSources` (or `genSourcesWithVineflower`) in `minecraft/mc262` to get local
decompiled 26.2 sources under `.gradle/loom-cache/minecraftMaven/.../*-sources.jar`, unzip them,
and grep for the actual current API before touching these files.
