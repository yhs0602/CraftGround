package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.AddListenerInterface;
import com.kyhsgeekcode.minecraftenv.EntityRenderListener;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.entity.state.EntityRenderState;
import net.minecraft.client.renderer.extract.LevelExtractor;
import net.minecraft.world.entity.Entity;
import org.jetbrains.annotations.NotNull;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

// 26.2 split WorldRenderer.render()/renderEntity() (mc121) into
// LevelExtractor.extractVisibleEntities()/extractEntity() (extract/render split, see
// docs/26_2_phase2_plan.md W8). This mixin merges what used to be two separate mc121 mixins
// (ClientWorldMixin + WorldRendererCallEntityRenderMixin) since both of their targets now live
// on the same class.
@Mixin(LevelExtractor.class)
public class LevelExtractorEntityListenerMixin implements AddListenerInterface {
  private final List<EntityRenderListener> listeners = new ArrayList<>();

  @Redirect(
      method = "extractVisibleEntities",
      at =
          @At(
              value = "INVOKE",
              target = "Lnet/minecraft/client/multiplayer/ClientLevel;entitiesForRendering()Ljava/lang/Iterable;"))
  private Iterable<Entity> getEntitiesForRendering(ClientLevel level) {
    for (EntityRenderListener listener : listeners) {
      listener.clear();
    }
    return level.entitiesForRendering();
  }

  @Inject(method = "extractEntity", at = @At("RETURN"))
  private void callOnEntityRender(
      Entity entity, float partialTickTime, CallbackInfoReturnable<EntityRenderState> cir) {
    for (EntityRenderListener listener : listeners) {
      listener.onEntityRender(entity);
    }
  }

  @Override
  public void addRenderListener(@NotNull EntityRenderListener listener) {
    listeners.add(listener);
  }

  @Override
  public List<EntityRenderListener> getRenderListeners() {
    return listeners;
  }
}
