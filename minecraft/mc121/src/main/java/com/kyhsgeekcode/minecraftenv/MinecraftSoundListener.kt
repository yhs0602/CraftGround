package com.kyhsgeekcode.minecraftenv

import com.kyhsgeekcode.minecraftenv.proto.soundEntry
import net.minecraft.client.sound.SoundInstance
import net.minecraft.client.sound.SoundInstanceListener
import net.minecraft.client.sound.SoundManager
import net.minecraft.client.sound.WeightedSoundSet
import net.minecraft.text.TranslatableTextContent

data class SoundEntry(
    val translateKey: String,
    var age: Long,
    var x: Double,
    var y: Double,
    var z: Double,
) {
    fun reset(
        x: Double,
        y: Double,
        z: Double,
    ) {
        this.x = x
        this.y = y
        this.z = z
        age = 0
    }

    fun toMessage() =
        soundEntry {
            translateKey = this@SoundEntry.translateKey
            age = this@SoundEntry.age
            x = this@SoundEntry.x
            y = this@SoundEntry.y
            z = this@SoundEntry.z
        }
}

class MinecraftSoundListener(
    soundManager: SoundManager,
) : SoundInstanceListener {
    init {
        soundManager.registerListener(this)
    }

    private val _entries: MutableList<SoundEntry> = mutableListOf()
    val entries = _entries as List<SoundEntry>

    override fun onSoundPlayed(
        sound: SoundInstance?,
        soundSet: WeightedSoundSet?,
        range: Float,
    ) {
        if (sound == null) return
        if (soundSet == null) return
        val subtitle = soundSet.subtitle ?: return
        val content = subtitle.content
        if (content !is TranslatableTextContent) {
            println("content is not TranslatableTextContent: $content")
            return
        }
        val translateKey = content.key
        if (this._entries.isNotEmpty()) {
            for (subtitleEntry in this._entries) {
                if (subtitleEntry.translateKey != translateKey) continue
                subtitleEntry.reset(sound.x, sound.y, sound.z)
                return
            }
        }
//        print("new subtitle: $translateKey")
        this._entries.add(SoundEntry(translateKey, 0, sound.x, sound.y, sound.z))
    }

    // remove old entries
    fun onTick() {
        val iterator: MutableIterator<SoundEntry> = _entries.iterator()
        while (iterator.hasNext()) {
            val subtitleEntry = iterator.next()
            subtitleEntry.age++
            if (subtitleEntry.age > 60) { // for 3 seconds = 60 ticks
                iterator.remove()
                continue
            }
        }
    }
}
