package com.kyhsgeekcode.minecraftenv.mixin;

import com.kyhsgeekcode.minecraftenv.AudioLoopbackCapturer;
import net.minecraft.client.sound.SoundEngine;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.openal.ALC10;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import static com.kyhsgeekcode.minecraftenv.PrintWithTimeKt.printWithTime;

/**
 * Mixin to initialize AudioLoopbackCapturer after SoundEngine is initialized.
 * 
 * This hooks into the SoundEngine.init() method to:
 * 1. Wait for the main OpenAL device to be created
 * 2. Initialize the loopback capture device with the same audio format
 * 3. Enable waveform capture for RL observations
 */
@Mixin(SoundEngine.class)
public class SoundEngineMixin {
    
    @Shadow
    private long devicePointer;
    
    /**
     * Inject after SoundEngine.init() completes successfully.
     * At this point, the main OpenAL device and context are ready.
     */
    @Inject(
        method = "init",
        at = @At("TAIL")
    )
    private void onSoundEngineInit(@Nullable String deviceSpecifier, boolean directionalAudio, CallbackInfo ci) {
        printWithTime("SoundEngineMixin: SoundEngine initialized, setting up audio loopback capture");
        
        try {
            // Check if we should enable audio loopback (can be controlled via environment variable)
            String enableLoopback = System.getenv("CRAFTGROUND_AUDIO_LOOPBACK");
            if (enableLoopback != null && enableLoopback.equals("0")) {
                printWithTime("SoundEngineMixin: Audio loopback disabled via CRAFTGROUND_AUDIO_LOOPBACK=0");
                return;
            }
            
            // Initialize the loopback capturer with the main device pointer
            boolean success = AudioLoopbackCapturer.INSTANCE.initialize(this.devicePointer);
            
            if (success) {
                printWithTime("SoundEngineMixin: Audio loopback capture initialized successfully");
                printWithTime("SoundEngineMixin: Sample rate: " + AudioLoopbackCapturer.INSTANCE.getSampleRate() + " Hz");
                printWithTime("SoundEngineMixin: Channels: " + AudioLoopbackCapturer.INSTANCE.getChannels());
                printWithTime("SoundEngineMixin: Bits per sample: " + AudioLoopbackCapturer.INSTANCE.getBitsPerSample());
            } else {
                printWithTime("SoundEngineMixin: Audio loopback capture not available (extension may not be supported)");
            }
            
        } catch (Exception e) {
            printWithTime("SoundEngineMixin: Failed to initialize audio loopback: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Inject before SoundEngine.close() to clean up loopback resources.
     */
    @Inject(
        method = "close",
        at = @At("HEAD")
    )
    private void onSoundEngineClose(CallbackInfo ci) {
        printWithTime("SoundEngineMixin: SoundEngine closing, cleaning up audio loopback");
        
        try {
            AudioLoopbackCapturer.INSTANCE.close();
        } catch (Exception e) {
            printWithTime("SoundEngineMixin: Error closing audio loopback: " + e.getMessage());
        }
    }
}
