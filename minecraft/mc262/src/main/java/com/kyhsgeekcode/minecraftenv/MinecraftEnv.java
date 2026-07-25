package com.kyhsgeekcode.minecraftenv;

import net.fabricmc.api.ModInitializer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MinecraftEnv implements ModInitializer {
	public static final String MOD_ID = "minecraftenv";

	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		LOGGER.info("CraftGround minecraftenv (mc262 skeleton) initialized");
	}
}
