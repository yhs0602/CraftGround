package com.kyhsgeekcode.minecraftenv

import java.util.concurrent.locks.Condition
import java.util.concurrent.locks.ReentrantLock

internal class TickSynchronizer {
    private val lock = ReentrantLock()
    private val clientActionApplied: Condition = lock.newCondition()
    private val serverTickCompleted: Condition = lock.newCondition()

    @Volatile
    private var terminating = false // 종료 상태 추적

    @Volatile
    private var isClientActionApplied = false // 클라이언트 액션 적용 여부

    @Volatile
    private var isServerTickCompleted = false // 서버 틱 완료 여부

    // 클라이언트에서 액션 적용 후 호출
    fun notifyServerTickStart() {
        lock.lock()
        try {
            isClientActionApplied = true // 액션이 적용되었다고 표시
            clientActionApplied.signalAll()
        } finally {
            lock.unlock()
        }
    }

    // 서버에서 틱 시작 전 대기
    fun waitForClientAction() {
        lock.lock()
        try {
            while (!terminating && !isClientActionApplied) {
                clientActionApplied.await()
            }
            isClientActionApplied = false // 대기 조건 재설정
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        } finally {
            lock.unlock()
        }
    }

    // 서버 틱 완료 후 클라이언트 관찰 시작을 알림
    fun notifyClientSendObservation() {
        lock.lock()
        try {
            isServerTickCompleted = true // 서버 틱이 완료되었다고 표시
            serverTickCompleted.signalAll()
        } finally {
            lock.unlock()
        }
    }

    // 클라이언트에서 서버 틱 완료 후 관찰 전송 대기
    fun waitForServerTickCompletion() {
        lock.lock()
        try {
            while (!terminating && !isServerTickCompleted) {
                serverTickCompleted.await()
            }
            isServerTickCompleted = false // 대기 조건 재설정
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        } finally {
            lock.unlock()
        }
    }

    // 종료 메소드
    fun terminate() {
        lock.lock()
        try {
            terminating = true
            clientActionApplied.signalAll() // 모든 대기 중인 스레드 깨우기
            serverTickCompleted.signalAll()
        } finally {
            lock.unlock()
        }
    }
}
