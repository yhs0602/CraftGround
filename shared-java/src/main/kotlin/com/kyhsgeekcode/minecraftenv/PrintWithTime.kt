package com.kyhsgeekcode.minecraftenv

import java.time.format.DateTimeFormatter

val printWithTimeFormatter: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSSSSS")
var doPrintWithTime = false

fun printWithTime(msg: String) {
    return
    if (doPrintWithTime) {
        println("${printWithTimeFormatter.format(java.time.LocalDateTime.now())} $msg")
    }
}

fun profileStartPrint(tag: String) {
    if (doPrintWithTime) {
        println("${printWithTimeFormatter.format(java.time.LocalDateTime.now())} start $tag")
    }
}

fun profileEndPrint(tag: String) {
    if (doPrintWithTime) {
        println("${printWithTimeFormatter.format(java.time.LocalDateTime.now())} end $tag")
    }
}
