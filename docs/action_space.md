# Action Space
Craftground provides two modes of action space: `ActionSpaceVersion.V1_MINEDOJO` and `ActionSpaceVersion.V2_MINERL_HUMAN`. Each mode has its own action space definition.

## `ActionSpaceVersion.V1_MINEDOJO`
- `ActionSpaceVersion.V1_MINEDOJO` is the action space definition, similar to the one used in [MineDojo](https://docs.minedojo.org/sections/core_api/action_space.html) project.

The action space is a list of integers, and each indices represent the following actions:

- `0`: 0: no-op, 1: Move forward, 2: Move backward
- `1`: 0: no-op, 1: Strafe right, 2: Strafe left
- `2`: 0: no-op, 1: Jump, 2: Sneak, 3: Sprint
- `3`: Camera delta pitch (0: -180, 24:180)
- `4`: Camera delta yaw (-180: 180)
- `5`: 0: no-op, 1: Use 2: Drop 3: Attack 4: Craft 5: Equip 6: Place 7: Destroy
- `6`: Argument for craft
- `7`: Argument for equip, place, and destroy

Currently the crafting action, argument for craft, equip, place, and destroy are not supported in Craftground.


## `ActionSpaceVersion.V2_MINERL_HUMAN`
- `ActionSpaceVersion.V2_MINERL_HUMAN` is the action space definition, similar to the one used in [Minerl 1.0.0+](https://minerl.readthedocs.io/en/latest/environments/index.html#action-space) project.

The action space is a dictionary with the following keys:

- `attack`: Attack the entity in front of the agent.
- `back`: Move the agent backward.
- `forward`: Move the agent forward.
- `jump`: Make the agent jump.
- `left`: Move the agent to the left.
- `right`: Move the agent to the right.
- `sneak`: Make the agent sneak.
- `sprint`: Make the agent sprint.
- `use`: Use the item in the agent's hand.
- `drop`: Drop the item in the agent's hand.
- `inventory`: Open the agent's inventory.
- `hotbar_1`: Select the first item in the agent's hotbar.
- ...
- `hotbar_9`: Select the ninth item in the agent's hotbar.
- `camera_pitch`: Change the agent's camera pitch.
- `camera_yaw`: Change the agent's camera yaw.


## `ActionSpaceMessage`

| Field    | Type            | Description               |
|----------|-----------------|---------------------------|
| action   | repeated int32  | Available player actions. |
| commands | repeated string | Any minecraft commands.   |


### How to execute minecraft command in a gymnasium wrapper?
```python
self.get_wrapper_attr("add_commands")(
    [
        f"setblock 1 2 3 minecraft:cake"
    ]
)
```