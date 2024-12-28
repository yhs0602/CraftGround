package com.kyhsgeekcode.minecraftenv.mixin;

import java.util.function.BooleanSupplier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.ServerTask;
import net.minecraft.server.ServerTickManager;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.TimeHelper;
import net.minecraft.util.Util;
import net.minecraft.util.profiler.Profiler;
import net.minecraft.util.thread.ReentrantThreadExecutor;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;

// https://github.com/gnembon/fabric-carpet/blob/master/src/main/java/carpet/mixins/MinecraftServer_tickspeedMixin.java
@Mixin(value = MinecraftServer.class, priority = Integer.MAX_VALUE - 10)
public abstract class MinecraftServer_tickspeedMixin extends ReentrantThreadExecutor<ServerTask> {
  @Shadow @Final private static Logger LOGGER;
  // just because profilerTimings class is public
  Pair<Long, Integer> profilerTimings = null;
  @Shadow private volatile boolean running;
  //    @Shadow
  //    private long timeReference;
  @Shadow private Profiler profiler;
  //    @Shadow
  //    private long nextTickTimestamp;
  @Shadow private volatile boolean loading;
  //    @Shadow
  //    private long lastTimeReference;
  @Shadow private boolean waitingForNextTick;
  @Shadow private int ticks;
  @Shadow private boolean needsDebugSetup;
  @Unique private float carpetMsptAccum = 0.0f;

  @Shadow private long lastOverloadWarningNanos;

  public MinecraftServer_tickspeedMixin(String name) {
    super(name);
  }

  @Shadow
  public abstract void tick(BooleanSupplier booleanSupplier_1);

  @Shadow
  protected abstract boolean shouldKeepTicking();

  @Shadow
  public abstract Iterable<ServerWorld> getWorlds();

  @Shadow
  protected abstract void runTasksTillTickEnd();

  @Shadow
  protected abstract void startTickMetrics();

  @Shadow
  protected abstract void endTickMetrics();

  @Shadow
  public abstract boolean isPaused();

  @Shadow
  public abstract ServerTickManager getTickManager();

  @Shadow private long tickStartTimeNanos = Util.getMeasuringTimeNano();

  @Shadow
  private static final long OVERLOAD_THRESHOLD_NANOS = 20L * TimeHelper.SECOND_IN_NANOS / 20L;

  @Shadow private long tickEndTimeNanos;

  @Shadow
  private void startTaskPerformanceLog() {}

  @Shadow
  private void pushPerformanceLogs() {}

  @Shadow
  private void pushFullTickLog() {}

  @Shadow private float averageTickTime;

  //    // Cancel the while statement
  //    @Redirect(method = "runServer", at = @At(value = "FIELD", target =
  // "Lnet/minecraft/server/MinecraftServer;running:Z"))
  //    private boolean cancelRunLoop(MinecraftServer server) {
  //        return false;
  //    } // target run()

  // Replaced the above cancelled while statement with this one
  // could possibly just inject that mspt selection at the beginning of the loop, but then adding
  // all mspt's to
  // replace 50L will be a hassle
  //    @Inject(method = "runServer", at = @At(value = "INVOKE", shift = At.Shift.AFTER,
  //            target =
  // "Lnet/minecraft/server/MinecraftServer;createMetadata()Lnet/minecraft/server/ServerMetadata;"))
  //    private void modifiedRunLoop(CallbackInfo ci) {
  ////        while (this.running) {
  ////            //long long_1 = Util.getMeasuringTimeMs() - this.timeReference;
  ////            //CM deciding on tick speed
  ////            long msThisTick = 0L;
  ////            long long_1 = 0L;
  ////
  ////            msThisTick = (long) carpetMsptAccum; // regular tick
  ////            carpetMsptAccum += TickSpeed.INSTANCE.getMspt() - msThisTick;
  ////
  ////            long_1 = Util.getMeasuringTimeMs() - this.timeReference;
  ////
  ////            //end tick deciding
  ////            //smoothed out delay to include mcpt component. With 50L gives defaults.
  ////            if (long_1 > /*2000L*/1000L + 20 * TickSpeed.INSTANCE.getMspt() &&
  // this.timeReference - this.lastTimeReference >= /*15000L*/10000L + 100 *
  // TickSpeed.INSTANCE.getMspt()) {
  ////                long long_2 = long_1 / TickSpeed.INSTANCE.getMspt();//50L;
  ////                LOGGER.warn("Can't keep up! Is the server overloaded? Running {}ms or {} ticks
  // behind", long_1, long_2);
  ////                this.timeReference += long_2 * TickSpeed.INSTANCE.getMspt();//50L;
  ////                this.lastTimeReference = this.timeReference;
  ////            }
  ////
  ////            if (this.needsDebugSetup) {
  ////                this.needsDebugSetup = false;
  ////                this.profilerTimings = Pair.of(Util.getMeasuringTimeNano(), ticks);
  ////                //this.field_33978 = new
  // MinecraftServer.class_6414(Util.getMeasuringTimeNano(), this.ticks);
  ////            }
  ////            this.timeReference += msThisTick;//50L;
  ////            //TickDurationMonitor tickDurationMonitor = TickDurationMonitor.create("Server");
  ////            //this.startMonitor(tickDurationMonitor);
  ////            this.startTickMetrics();
  ////            this.profiler.push("tick");
  ////            this.tick(this::shouldKeepTicking);
  ////            this.profiler.swap("nextTickWait");
  ////            while (this.runEveryTask()) {
  ////                Thread.yield();
  ////            } // ?
  ////            this.waitingForNextTick = true;
  ////            this.nextTickTimestamp = Math.max(Util.getMeasuringTimeMs() + /*50L*/ msThisTick,
  // this.timeReference);
  ////            // run all tasks (this will not do a lot when warping), but that's fine since we
  // already run them
  ////            this.runTasksTillTickEnd();//            this.runTasks();
  //////            this.runTasks();
  ////            this.profiler.pop();
  ////            this.endTickMetrics();
  ////            this.loading = true;
  //
  //            boolean bl;
  //            long l;
  //            var tickManager = this.getTickManager();
  //            if (!this.isPaused() && tickManager.isSprinting() && tickManager.sprint()) {
  //                l = 0L;
  //                this.lastOverloadWarningNanos = this.tickStartTimeNanos =
  // Util.getMeasuringTimeNano();
  //            } else {
  //                l = tickManager.getNanosPerTick();
  //                long m = Util.getMeasuringTimeNano() - this.tickStartTimeNanos;
  //                if (m > OVERLOAD_THRESHOLD_NANOS + 20L * l && this.tickStartTimeNanos -
  // this.lastOverloadWarningNanos >= OVERLOAD_WARNING_INTERVAL_NANOS + 100L * l) {
  //                    long n = m / l;
  //                    LOGGER.warn("Can't keep up! Is the server overloaded? Running {}ms or {}
  // ticks behind", (Object) (m / TimeHelper.MILLI_IN_NANOS), (Object) n);
  //                    this.tickStartTimeNanos += n * l;
  //                    this.lastOverloadWarningNanos = this.tickStartTimeNanos;
  //                }
  //            }
  //            boolean bl2 = bl = l == 0L;
  //            if (this.needsDebugSetup) {
  //                this.needsDebugSetup = false;
  //            }
  //            this.tickStartTimeNanos += l;
  //            this.startTickMetrics();
  //            this.profiler.push("tick");
  //            this.tick(bl ? () -> false : this::shouldKeepTicking);
  //            this.profiler.swap("nextTickWait");
  //            this.waitingForNextTick = true;
  //            this.tickEndTimeNanos = Math.max(Util.getMeasuringTimeNano() + l,
  // this.tickStartTimeNanos);
  //            this.startTaskPerformanceLog();
  //            this.runTasksTillTickEnd();
  //            this.pushPerformanceLogs();
  //            if (bl) {
  //                tickManager.updateSprintTime();
  //            }
  //            this.profiler.pop();
  //            this.pushFullTickLog();
  //            this.endTickMetrics();
  //            this.loading = true;
  //            FlightProfiler.INSTANCE.onTick(this.averageTickTime);
  //        }
  //    }

  //    private boolean runEveryTask() {
  //        if (super.runTask()) {
  //            return true;
  //        } else {
  //            if (true) { // unconditionally this time
  //                for (ServerWorld serverlevel : getWorlds()) {
  //                    if (serverlevel.getChunkManager().executeQueuedTasks()) {
  //                        return true;
  //                    }
  //                }
  //            }
  //            return false;
  //        }
  //    }
}
