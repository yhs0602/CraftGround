package com.kyhsgeekcode.minecraftenv;

import java.util.ArrayList;
import java.util.List;
import org.jetbrains.annotations.NotNull;

public interface AddListenerInterface {
  List<EntityRenderListener> listeners = new ArrayList<>();

  default void addRenderListener(@NotNull EntityRenderListener listener) {
    listeners.add(listener);
  }

  default List<EntityRenderListener> getRenderListeners() {
    return listeners;
  }
}
