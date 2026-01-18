package com.kyhsgeekcode.minecraftenv

import com.google.protobuf.ByteString
import org.lwjgl.openal.ALC10
import org.lwjgl.openal.ALC11
import org.lwjgl.openal.ALCCapabilities
import org.lwjgl.openal.SOFTLoopback
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import java.nio.ByteBuffer
import java.nio.ShortBuffer

/**
 * AudioLoopbackCapturer - Captures audio waveform from OpenAL using SOFT_loopback extension.
 * 
 * This class creates a secondary loopback device that mirrors the audio output,
 * allowing the RL agent to receive raw PCM waveform data.
 * 
 * Architecture:
 * - Main device: Normal audio output to speakers/headphones
 * - Loopback device: Captures rendered audio to memory buffer
 * 
 * Usage:
 * 1. Call initialize() after SoundEngine is initialized
 * 2. Call renderSamples() each tick to capture audio
 * 3. Call getWaveformData() to retrieve PCM data for transmission
 * 4. Call close() on shutdown
 */
object AudioLoopbackCapturer {
    private const val TAG = "AudioLoopbackCapturer"
    
    // Audio format settings
    const val SAMPLE_RATE = 44100  // Hz
    const val CHANNELS = 2         // Stereo
    const val BITS_PER_SAMPLE = 16 // 16-bit PCM
    
    // Buffer settings - capture ~100ms of audio per tick (at 20 ticks/sec = ~50ms per tick)
    // 44100 * 2 channels * 2 bytes * 0.05 sec = 8820 bytes per tick
    private const val SAMPLES_PER_RENDER = 2205  // ~50ms at 44100Hz
    private const val BUFFER_SIZE_BYTES = SAMPLES_PER_RENDER * CHANNELS * (BITS_PER_SAMPLE / 8)
    
    // OpenAL loopback device and context
    private var loopbackDevice: Long = 0L
    private var loopbackContext: Long = 0L
    
    // Audio buffer for capturing samples
    private var audioBuffer: ShortBuffer? = null
    private var lastCapturedSamples: ByteArray = ByteArray(0)
    
    // State flags
    private var isInitialized = false
    private var isLoopbackSupported = false
    
    /**
     * Check if ALC_SOFT_loopback extension is available
     */
    fun checkLoopbackSupport(device: Long): Boolean {
        return try {
            ALC10.alcIsExtensionPresent(device, "ALC_SOFT_loopback")
        } catch (e: Exception) {
            printWithTime("$TAG: Failed to check loopback support: ${e.message}")
            false
        }
    }
    
    /**
     * Initialize the loopback capture device.
     * Should be called after the main SoundEngine is initialized.
     * 
     * @param mainDevice The main OpenAL device pointer (for checking extension support)
     * @return true if initialization was successful
     */
    fun initialize(mainDevice: Long): Boolean {
        if (isInitialized) {
            printWithTime("$TAG: Already initialized")
            return true
        }
        
        try {
            // Check if loopback extension is supported
            isLoopbackSupported = checkLoopbackSupport(mainDevice)
            if (!isLoopbackSupported) {
                printWithTime("$TAG: ALC_SOFT_loopback extension not supported")
                return false
            }
            
            printWithTime("$TAG: ALC_SOFT_loopback extension is supported")
            
            // Open loopback device
            loopbackDevice = SOFTLoopback.alcLoopbackOpenDeviceSOFT(null as CharSequence?)
            if (loopbackDevice == 0L) {
                printWithTime("$TAG: Failed to open loopback device")
                return false
            }
            
            printWithTime("$TAG: Loopback device opened: $loopbackDevice")
            
            // Create context with specific audio format attributes
            MemoryStack.stackPush().use { stack ->
                val attrs = stack.mallocInt(9)
                attrs.put(ALC10.ALC_FREQUENCY).put(SAMPLE_RATE)
                attrs.put(SOFTLoopback.ALC_FORMAT_CHANNELS_SOFT).put(SOFTLoopback.ALC_STEREO_SOFT)
                attrs.put(SOFTLoopback.ALC_FORMAT_TYPE_SOFT).put(SOFTLoopback.ALC_SHORT_SOFT)
                attrs.put(0) // Terminate list
                attrs.flip()
                
                loopbackContext = ALC10.alcCreateContext(loopbackDevice, attrs)
            }
            
            if (loopbackContext == 0L) {
                printWithTime("$TAG: Failed to create loopback context")
                ALC10.alcCloseDevice(loopbackDevice)
                loopbackDevice = 0L
                return false
            }
            
            printWithTime("$TAG: Loopback context created: $loopbackContext")
            
            // Allocate audio buffer
            audioBuffer = MemoryUtil.memAllocShort(SAMPLES_PER_RENDER * CHANNELS)
            
            isInitialized = true
            printWithTime("$TAG: Initialization complete - Sample Rate: $SAMPLE_RATE Hz, Channels: $CHANNELS, Buffer: ${BUFFER_SIZE_BYTES} bytes")
            
            return true
            
        } catch (e: Exception) {
            printWithTime("$TAG: Initialization failed: ${e.message}")
            e.printStackTrace()
            close()
            return false
        }
    }
    
    /**
     * Render audio samples from the loopback device.
     * Should be called each tick to capture the latest audio.
     * 
     * Note: This requires the loopback context to be current when sources are playing.
     * In practice, this means we need to either:
     * 1. Use a shared context approach
     * 2. Mirror all sound commands to the loopback device
     * 3. Use OS-level audio loopback as fallback
     * 
     * For now, this demonstrates the API usage. Full integration requires
     * mirroring the SoundSystem's source operations to the loopback context.
     */
    fun renderSamples(): Boolean {
        if (!isInitialized || !isLoopbackSupported) {
            return false
        }
        
        val buffer = audioBuffer ?: return false
        
        try {
            // Make loopback context current for rendering
            val previousContext = ALC10.alcGetCurrentContext()
            ALC10.alcMakeContextCurrent(loopbackContext)
            
            // Clear buffer
            buffer.clear()
            
            // Render samples from all active sources in the loopback context
            SOFTLoopback.alcRenderSamplesSOFT(loopbackDevice, buffer, SAMPLES_PER_RENDER)
            
            // Restore previous context
            ALC10.alcMakeContextCurrent(previousContext)
            
            // Convert to byte array for transmission
            buffer.rewind()
            lastCapturedSamples = ByteArray(BUFFER_SIZE_BYTES)
            for (i in 0 until SAMPLES_PER_RENDER * CHANNELS) {
                val sample = buffer.get(i)
                lastCapturedSamples[i * 2] = (sample.toInt() and 0xFF).toByte()
                lastCapturedSamples[i * 2 + 1] = ((sample.toInt() shr 8) and 0xFF).toByte()
            }
            
            return true
            
        } catch (e: Exception) {
            printWithTime("$TAG: Failed to render samples: ${e.message}")
            return false
        }
    }
    
    /**
     * Get the last captured waveform data as a Protobuf ByteString.
     */
    fun getWaveformData(): ByteString {
        return ByteString.copyFrom(lastCapturedSamples)
    }
    
    /**
     * Get the last captured waveform data as a raw byte array.
     */
    fun getWaveformBytes(): ByteArray {
        return lastCapturedSamples
    }
    
    /**
     * Get audio format information for the Python client.
     */
    fun getSampleRate(): Int = SAMPLE_RATE
    fun getChannels(): Int = CHANNELS
    fun getBitsPerSample(): Int = BITS_PER_SAMPLE
    fun getSamplesPerRender(): Int = SAMPLES_PER_RENDER
    
    /**
     * Check if loopback capture is enabled and working.
     */
    fun isEnabled(): Boolean = isInitialized && isLoopbackSupported
    
    /**
     * Clean up resources.
     */
    fun close() {
        if (loopbackContext != 0L) {
            ALC10.alcDestroyContext(loopbackContext)
            loopbackContext = 0L
        }
        
        if (loopbackDevice != 0L) {
            ALC10.alcCloseDevice(loopbackDevice)
            loopbackDevice = 0L
        }
        
        audioBuffer?.let {
            MemoryUtil.memFree(it)
            audioBuffer = null
        }
        
        lastCapturedSamples = ByteArray(0)
        isInitialized = false
        
        printWithTime("$TAG: Closed")
    }
}
