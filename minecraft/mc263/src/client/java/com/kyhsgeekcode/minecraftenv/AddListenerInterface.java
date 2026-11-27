package com.kyhsgeekcode.minecraftenv;

import java.util.List;
import org.jetbrains.annotations.NotNull;

public interface AddListenerInterface {
  void addRenderListener(@NotNull EntityRenderListener listener);

  List<EntityRenderListener> getRenderListeners();
}
