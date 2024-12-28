package com.kyhsgeekcode.minecraftenv

import net.minecraft.client.texture.NativeImage
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import javax.imageio.ImageIO

@Throws(IOException::class)
fun encodeImageToBytes(
    image: NativeImage,
    originalSizeX: Int,
    originalSizeY: Int,
    targetSizeX: Int,
    targetSizeY: Int,
): ByteArray {
//    if (originalSizeX == targetSizeX && originalSizeY == targetSizeY)
//        return image.bytes
    val data = image.bytes
    val originalImage = ImageIO.read(ByteArrayInputStream(data))
    val resizedImage = BufferedImage(targetSizeX, targetSizeY, originalImage.type)
    val graphics = resizedImage.createGraphics()
    graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    graphics.drawImage(originalImage, 0, 0, targetSizeX, targetSizeY, null)
    graphics.dispose()
    val baos = ByteArrayOutputStream()
    ImageIO.write(resizedImage, "png", baos)
    return baos.toByteArray()
}
