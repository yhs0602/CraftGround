package com.kyhsgeekcode.minecraftenv.client;

import net.fabricmc.api.ClientModInitializer;

import com.kyhsgeekcode.minecraftenv.MinecraftEnv;

public class MinecraftEnvClient implements ClientModInitializer {
	@Override
	public void onInitializeClient() {
		MinecraftEnv.LOGGER.info("CraftGround minecraftenv client (mc262 skeleton) initialized");
	}
}
