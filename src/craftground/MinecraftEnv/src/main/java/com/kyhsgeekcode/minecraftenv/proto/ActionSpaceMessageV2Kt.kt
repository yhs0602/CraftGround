// Generated by the protocol buffer compiler. DO NOT EDIT!
// NO CHECKED-IN PROTOBUF GENCODE
// source: proto/action_space.proto

// Generated files should ignore deprecation warnings
@file:Suppress("DEPRECATION")
package com.kyhsgeekcode.minecraftenv.proto;

@kotlin.jvm.JvmName("-initializeactionSpaceMessageV2")
public inline fun actionSpaceMessageV2(block: com.kyhsgeekcode.minecraftenv.proto.ActionSpaceMessageV2Kt.Dsl.() -> kotlin.Unit): com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2 =
  com.kyhsgeekcode.minecraftenv.proto.ActionSpaceMessageV2Kt.Dsl._create(com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2.newBuilder()).apply { block() }._build()
/**
 * Protobuf type `ActionSpaceMessageV2`
 */
public object ActionSpaceMessageV2Kt {
  @kotlin.OptIn(com.google.protobuf.kotlin.OnlyForUseByGeneratedProtoCode::class)
  @com.google.protobuf.kotlin.ProtoDslMarker
  public class Dsl private constructor(
    private val _builder: com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2.Builder
  ) {
    public companion object {
      @kotlin.jvm.JvmSynthetic
    @kotlin.PublishedApi
      internal fun _create(builder: com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2.Builder): Dsl = Dsl(builder)
    }

    @kotlin.jvm.JvmSynthetic
  @kotlin.PublishedApi
    internal fun _build(): com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2 = _builder.build()

    /**
     * ```
     * Discrete actions for movement and other commands as bool
     * ```
     *
     * `bool attack = 1;`
     */
    public var attack: kotlin.Boolean
      @JvmName("getAttack")
      get() = _builder.attack
      @JvmName("setAttack")
      set(value) {
        _builder.attack = value
      }
    /**
     * ```
     * Discrete actions for movement and other commands as bool
     * ```
     *
     * `bool attack = 1;`
     */
    public fun clearAttack() {
      _builder.clearAttack()
    }

    /**
     * `bool back = 2;`
     */
    public var back: kotlin.Boolean
      @JvmName("getBack")
      get() = _builder.back
      @JvmName("setBack")
      set(value) {
        _builder.back = value
      }
    /**
     * `bool back = 2;`
     */
    public fun clearBack() {
      _builder.clearBack()
    }

    /**
     * `bool forward = 3;`
     */
    public var forward: kotlin.Boolean
      @JvmName("getForward")
      get() = _builder.forward
      @JvmName("setForward")
      set(value) {
        _builder.forward = value
      }
    /**
     * `bool forward = 3;`
     */
    public fun clearForward() {
      _builder.clearForward()
    }

    /**
     * `bool jump = 4;`
     */
    public var jump: kotlin.Boolean
      @JvmName("getJump")
      get() = _builder.jump
      @JvmName("setJump")
      set(value) {
        _builder.jump = value
      }
    /**
     * `bool jump = 4;`
     */
    public fun clearJump() {
      _builder.clearJump()
    }

    /**
     * `bool left = 5;`
     */
    public var left: kotlin.Boolean
      @JvmName("getLeft")
      get() = _builder.left
      @JvmName("setLeft")
      set(value) {
        _builder.left = value
      }
    /**
     * `bool left = 5;`
     */
    public fun clearLeft() {
      _builder.clearLeft()
    }

    /**
     * `bool right = 6;`
     */
    public var right: kotlin.Boolean
      @JvmName("getRight")
      get() = _builder.right
      @JvmName("setRight")
      set(value) {
        _builder.right = value
      }
    /**
     * `bool right = 6;`
     */
    public fun clearRight() {
      _builder.clearRight()
    }

    /**
     * `bool sneak = 7;`
     */
    public var sneak: kotlin.Boolean
      @JvmName("getSneak")
      get() = _builder.sneak
      @JvmName("setSneak")
      set(value) {
        _builder.sneak = value
      }
    /**
     * `bool sneak = 7;`
     */
    public fun clearSneak() {
      _builder.clearSneak()
    }

    /**
     * `bool sprint = 8;`
     */
    public var sprint: kotlin.Boolean
      @JvmName("getSprint")
      get() = _builder.sprint
      @JvmName("setSprint")
      set(value) {
        _builder.sprint = value
      }
    /**
     * `bool sprint = 8;`
     */
    public fun clearSprint() {
      _builder.clearSprint()
    }

    /**
     * `bool use = 9;`
     */
    public var use: kotlin.Boolean
      @JvmName("getUse")
      get() = _builder.use
      @JvmName("setUse")
      set(value) {
        _builder.use = value
      }
    /**
     * `bool use = 9;`
     */
    public fun clearUse() {
      _builder.clearUse()
    }

    /**
     * `bool drop = 10;`
     */
    public var drop: kotlin.Boolean
      @JvmName("getDrop")
      get() = _builder.drop
      @JvmName("setDrop")
      set(value) {
        _builder.drop = value
      }
    /**
     * `bool drop = 10;`
     */
    public fun clearDrop() {
      _builder.clearDrop()
    }

    /**
     * `bool inventory = 11;`
     */
    public var inventory: kotlin.Boolean
      @JvmName("getInventory")
      get() = _builder.inventory
      @JvmName("setInventory")
      set(value) {
        _builder.inventory = value
      }
    /**
     * `bool inventory = 11;`
     */
    public fun clearInventory() {
      _builder.clearInventory()
    }

    /**
     * ```
     * Hotbar selection (1-9) as bool
     * ```
     *
     * `bool hotbar_1 = 12;`
     */
    public var hotbar1: kotlin.Boolean
      @JvmName("getHotbar1")
      get() = _builder.hotbar1
      @JvmName("setHotbar1")
      set(value) {
        _builder.hotbar1 = value
      }
    /**
     * ```
     * Hotbar selection (1-9) as bool
     * ```
     *
     * `bool hotbar_1 = 12;`
     */
    public fun clearHotbar1() {
      _builder.clearHotbar1()
    }

    /**
     * `bool hotbar_2 = 13;`
     */
    public var hotbar2: kotlin.Boolean
      @JvmName("getHotbar2")
      get() = _builder.hotbar2
      @JvmName("setHotbar2")
      set(value) {
        _builder.hotbar2 = value
      }
    /**
     * `bool hotbar_2 = 13;`
     */
    public fun clearHotbar2() {
      _builder.clearHotbar2()
    }

    /**
     * `bool hotbar_3 = 14;`
     */
    public var hotbar3: kotlin.Boolean
      @JvmName("getHotbar3")
      get() = _builder.hotbar3
      @JvmName("setHotbar3")
      set(value) {
        _builder.hotbar3 = value
      }
    /**
     * `bool hotbar_3 = 14;`
     */
    public fun clearHotbar3() {
      _builder.clearHotbar3()
    }

    /**
     * `bool hotbar_4 = 15;`
     */
    public var hotbar4: kotlin.Boolean
      @JvmName("getHotbar4")
      get() = _builder.hotbar4
      @JvmName("setHotbar4")
      set(value) {
        _builder.hotbar4 = value
      }
    /**
     * `bool hotbar_4 = 15;`
     */
    public fun clearHotbar4() {
      _builder.clearHotbar4()
    }

    /**
     * `bool hotbar_5 = 16;`
     */
    public var hotbar5: kotlin.Boolean
      @JvmName("getHotbar5")
      get() = _builder.hotbar5
      @JvmName("setHotbar5")
      set(value) {
        _builder.hotbar5 = value
      }
    /**
     * `bool hotbar_5 = 16;`
     */
    public fun clearHotbar5() {
      _builder.clearHotbar5()
    }

    /**
     * `bool hotbar_6 = 17;`
     */
    public var hotbar6: kotlin.Boolean
      @JvmName("getHotbar6")
      get() = _builder.hotbar6
      @JvmName("setHotbar6")
      set(value) {
        _builder.hotbar6 = value
      }
    /**
     * `bool hotbar_6 = 17;`
     */
    public fun clearHotbar6() {
      _builder.clearHotbar6()
    }

    /**
     * `bool hotbar_7 = 18;`
     */
    public var hotbar7: kotlin.Boolean
      @JvmName("getHotbar7")
      get() = _builder.hotbar7
      @JvmName("setHotbar7")
      set(value) {
        _builder.hotbar7 = value
      }
    /**
     * `bool hotbar_7 = 18;`
     */
    public fun clearHotbar7() {
      _builder.clearHotbar7()
    }

    /**
     * `bool hotbar_8 = 19;`
     */
    public var hotbar8: kotlin.Boolean
      @JvmName("getHotbar8")
      get() = _builder.hotbar8
      @JvmName("setHotbar8")
      set(value) {
        _builder.hotbar8 = value
      }
    /**
     * `bool hotbar_8 = 19;`
     */
    public fun clearHotbar8() {
      _builder.clearHotbar8()
    }

    /**
     * `bool hotbar_9 = 20;`
     */
    public var hotbar9: kotlin.Boolean
      @JvmName("getHotbar9")
      get() = _builder.hotbar9
      @JvmName("setHotbar9")
      set(value) {
        _builder.hotbar9 = value
      }
    /**
     * `bool hotbar_9 = 20;`
     */
    public fun clearHotbar9() {
      _builder.clearHotbar9()
    }

    /**
     * ```
     * Camera movement (pitch and yaw)
     * ```
     *
     * `float camera_pitch = 21;`
     */
    public var cameraPitch: kotlin.Float
      @JvmName("getCameraPitch")
      get() = _builder.cameraPitch
      @JvmName("setCameraPitch")
      set(value) {
        _builder.cameraPitch = value
      }
    /**
     * ```
     * Camera movement (pitch and yaw)
     * ```
     *
     * `float camera_pitch = 21;`
     */
    public fun clearCameraPitch() {
      _builder.clearCameraPitch()
    }

    /**
     * `float camera_yaw = 22;`
     */
    public var cameraYaw: kotlin.Float
      @JvmName("getCameraYaw")
      get() = _builder.cameraYaw
      @JvmName("setCameraYaw")
      set(value) {
        _builder.cameraYaw = value
      }
    /**
     * `float camera_yaw = 22;`
     */
    public fun clearCameraYaw() {
      _builder.clearCameraYaw()
    }

    /**
     * An uninstantiable, behaviorless type to represent the field in
     * generics.
     */
    @kotlin.OptIn(com.google.protobuf.kotlin.OnlyForUseByGeneratedProtoCode::class)
    public class CommandsProxy private constructor() : com.google.protobuf.kotlin.DslProxy()
    /**
     * `repeated string commands = 23;`
     * @return A list containing the commands.
     */
    public val commands: com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>
      @kotlin.jvm.JvmSynthetic
      get() = com.google.protobuf.kotlin.DslList(
        _builder.commandsList
      )
    /**
     * `repeated string commands = 23;`
     * @param value The commands to add.
     */
    @kotlin.jvm.JvmSynthetic
    @kotlin.jvm.JvmName("addCommands")
    public fun com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>.add(value: kotlin.String) {
      _builder.addCommands(value)
    }
    /**
     * `repeated string commands = 23;`
     * @param value The commands to add.
     */
    @kotlin.jvm.JvmSynthetic
    @kotlin.jvm.JvmName("plusAssignCommands")
    @Suppress("NOTHING_TO_INLINE")
    public inline operator fun com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>.plusAssign(value: kotlin.String) {
      add(value)
    }
    /**
     * `repeated string commands = 23;`
     * @param values The commands to add.
     */
    @kotlin.jvm.JvmSynthetic
    @kotlin.jvm.JvmName("addAllCommands")
    public fun com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>.addAll(values: kotlin.collections.Iterable<kotlin.String>) {
      _builder.addAllCommands(values)
    }
    /**
     * `repeated string commands = 23;`
     * @param values The commands to add.
     */
    @kotlin.jvm.JvmSynthetic
    @kotlin.jvm.JvmName("plusAssignAllCommands")
    @Suppress("NOTHING_TO_INLINE")
    public inline operator fun com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>.plusAssign(values: kotlin.collections.Iterable<kotlin.String>) {
      addAll(values)
    }
    /**
     * `repeated string commands = 23;`
     * @param index The index to set the value at.
     * @param value The commands to set.
     */
    @kotlin.jvm.JvmSynthetic
    @kotlin.jvm.JvmName("setCommands")
    public operator fun com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>.set(index: kotlin.Int, value: kotlin.String) {
      _builder.setCommands(index, value)
    }/**
     * `repeated string commands = 23;`
     */
    @kotlin.jvm.JvmSynthetic
    @kotlin.jvm.JvmName("clearCommands")
    public fun com.google.protobuf.kotlin.DslList<kotlin.String, CommandsProxy>.clear() {
      _builder.clearCommands()
    }}
}
@kotlin.jvm.JvmSynthetic
public inline fun com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2.copy(block: `com.kyhsgeekcode.minecraftenv.proto`.ActionSpaceMessageV2Kt.Dsl.() -> kotlin.Unit): com.kyhsgeekcode.minecraftenv.proto.ActionSpace.ActionSpaceMessageV2 =
  `com.kyhsgeekcode.minecraftenv.proto`.ActionSpaceMessageV2Kt.Dsl._create(this.toBuilder()).apply { block() }._build()

