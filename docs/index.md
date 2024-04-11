# CraftGround

A **fast**, **up-to-date**, and **feature-rich** Minecraft-based reinforcement learning environment.

## Version

- Supports Minecraft version `1.19.4`.
- Current version of CraftGround: `1.7.23`.

## Features

### Initial Environment

Refer to the below proto for the InitialEnvironment:

```proto
message InitialEnvironmentMessage {
  repeated string initialInventoryCommands = 1;
  repeated int32 initialPosition = 2;
  repeated string initialMobsCommands = 3;
  int32 imageSizeX = 4;
  int32 imageSizeY = 5;
  int64 seed = 6;
  bool allowMobSpawn = 7;
  bool alwaysNight = 8;
  bool alwaysDay = 9;
  string initialWeather = 10;
  bool isWorldFlat = 11;
  int32 visibleSizeX = 12;
  int32 visibleSizeY = 13;
  repeated string initialExtraCommands = 14;
  repeated string killedStatKeys = 15;
  repeated string minedStatKeys = 16;
  repeated string miscStatKeys = 17;
  repeated BlockState initialBlockStates = 18;
  repeated int32 surroundingEntityDistances = 19;
  bool hudHidden = 20;
  int32 render_distance = 21;
  int32 simulation_distance = 22;
  bool biocular = 23;
  float eye_distance = 24;
  repeated string structurePaths = 25;
  bool noWeatherCycle = 26;
  bool no_pov_effect = 27;
  bool noTimeCycle = 28;
  bool request_raycast = 29;
  int32 screen_encoding_mode = 30;
}
```

### Observation Space
Includes basic vision rendering, binocular rendering, list of sounds around the agent, agent's status effects, and more. See the proto file for detailed information.

```proto
message ItemStack {
  int32 raw_id = 1;
  string translation_key = 2;
  int32 count = 3;
  int32 durability = 4;
  int32 max_durability = 5;
}

message BlockInfo {
  int32 x = 1;
  int32 y = 2;
  int32 z = 3;
  string translation_key = 4;
}

message EntityInfo {
  string unique_name = 1;
  string translation_key = 2;
  double x = 3;
  double y = 4;
  double z = 5;
  double yaw = 6;
  double pitch = 7;
  double health = 8;
}

message HitResult {
  enum Type {
    MISS = 0;
    BLOCK = 1;
    ENTITY = 2;
  }

  Type type = 1;
  BlockInfo target_block = 2;
  EntityInfo target_entity = 3;
}

message StatusEffect {
  string translation_key = 1;
  int32 duration = 2;
  int32 amplifier = 3;
}

message SoundEntry {
  string translate_key = 1;
  int64 age = 2;
  double x = 3;
  double y = 4;
  double z = 5;
}

message EntitiesWithinDistance {
  repeated EntityInfo entities = 1;
}

message ObservationSpaceMessage {
  bytes image = 1;
  double x = 2;
  double y = 3;
  double z = 4;
  double yaw = 5;
  double pitch = 6;
  double health = 7;
  double food_level = 8;
  double saturation_level = 9;
  bool is_dead = 10;
  repeated ItemStack inventory = 11;
  HitResult raycast_result = 12;
  repeated SoundEntry sound_subtitles = 13;
  repeated StatusEffect status_effects = 14;
  map<string, int32> killed_statistics = 15;
  map<string, int32> mined_statistics = 16;
  map<string, int32> misc_statistics = 17;
  repeated EntityInfo visible_entities = 18;
  map<int32, EntitiesWithinDistance> surrounding_entities = 19;
  bool bobber_thrown = 20;
  int32 experience = 21;
  int64 world_time = 22;
  string last_death_message = 23;
  bytes image_2 = 24;
}
```
### Action Space
Similar to Minedojo. (Crafting Not supported)
```proto
message ActionSpaceMessage {
  repeated int32 action = 1;
  repeated string commands = 2;
}
```

## Headless Server Support
Supports headless offscreen rendering using VirtualGL and Xvfb.

## Installation
```shell
pip install git+https://github.com/yhs0602/CraftGround
```
- Dependencies: JDK 17, OpenGL, GLEW, libpng, zlib

## Technical Report
Refer to the [Technical Report](https://yhs0602.github.io/CraftGround/technical_report) for detailed information on CraftGround's internals, optimizations, and more.