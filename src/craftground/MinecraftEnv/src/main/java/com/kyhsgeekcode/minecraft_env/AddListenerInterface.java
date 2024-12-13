package com.kyhsgeekcode.minecraft_env;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public interface AddListenerInterface {
    List<EntityRenderListener> listeners = new ArrayList<>();

    default void addRenderListener(@NotNull EntityRenderListener listener) {
        listeners.add(listener);
    }

    default List<EntityRenderListener> getRenderListeners() {
        return listeners;
    }
}
