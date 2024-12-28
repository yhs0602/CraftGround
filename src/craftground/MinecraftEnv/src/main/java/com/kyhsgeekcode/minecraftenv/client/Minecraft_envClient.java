package com.kyhsgeekcode.minecraftenv.client;

import net.fabricmc.api.ClientModInitializer;

public class Minecraft_envClient implements ClientModInitializer {
  @Override
  public void onInitializeClient() {
    System.out.println("Hello Fabric world! client");
    var ld_preload = System.getenv("LD_PRELOAD");
    if (ld_preload != null) {
      System.out.println("LD_PRELOAD is set: " + ld_preload);
    } else {
      System.out.println("LD_PRELOAD is not set");
    }
  }
}
