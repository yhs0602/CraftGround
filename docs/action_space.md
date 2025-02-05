# **Action Space**  

Craftground provides two modes of action space:  
- **`ActionSpaceVersion.V1_MINEDOJO`** – Similar to the action space used in [MineDojo](https://docs.minedojo.org/sections/core_api/action_space.html).  
- **`ActionSpaceVersion.V2_MINERL_HUMAN`** – Similar to the action space in [MineRL 1.0.0+](https://minerl.readthedocs.io/en/latest/environments/index.html#action-space).  

Each mode defines a different representation of agent actions.  

---

## **`ActionSpaceVersion.V1_MINEDOJO`**  

This action space follows the MineDojo convention and represents actions as a list of integers, with each index corresponding to a specific action category.  

### **Action Space Structure**

| Index | Action                      | Values                                                                                         |
| ----- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| `0`   | Movement (forward/backward) | `0`: No-op, `1`: Move forward, `2`: Move backward                                              |
| `1`   | Strafing (left/right)       | `0`: No-op, `1`: Strafe right, `2`: Strafe left                                                |
| `2`   | Movement modifiers          | `0`: No-op, `1`: Jump, `2`: Sneak, `3`: Sprint                                                 |
| `3`   | Camera pitch adjustment     | `0`: -180° to `24`: 180°                                                                       |
| `4`   | Camera yaw adjustment       | `-180` to `180`                                                                                |
| `5`   | Interaction                 | `0`: No-op, `1`: Use, `2`: Drop, `3`: Attack, `4`: Craft, `5`: Equip, `6`: Place, `7`: Destroy |
| `6`   | Crafting argument           | Argument for `craft` (not yet supported)                                                       |
| `7`   | Item manipulation           | Argument for `equip`, `place`, and `destroy` (not yet supported)                               |


### **Limitations**  
- Currently, `craft`, `equip`, `place`, and `destroy` actions are **not supported** in Craftground.  

---

## **`ActionSpaceVersion.V2_MINERL_HUMAN`**  

This action space follows the MineRL convention and represents actions using a dictionary format. Each action is mapped to a discrete or continuous value.  

### **Action Space Structure**
  
| Key                     | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `attack`                | Attack an entity in front of the agent.          |
| `back`                  | Move the agent backward.                         |
| `forward`               | Move the agent forward.                          |
| `jump`                  | Make the agent jump.                             |
| `left`                  | Move the agent to the left.                      |
| `right`                 | Move the agent to the right.                     |
| `sneak`                 | Make the agent sneak.                            |
| `sprint`                | Make the agent sprint.                           |
| `use`                   | Use the item in the agent’s hand.                |
| `drop`                  | Drop the item in the agent’s hand.               |
| `inventory`             | Open the agent’s inventory.                      |
| `hotbar_1` - `hotbar_9` | Select an item slot in the agent’s hotbar (1-9). |
| `camera_pitch`          | Adjust the agent’s camera pitch.                 |
| `camera_yaw`            | Adjust the agent’s camera yaw.                   |

This version is closer to human-like control and is better suited for reinforcement learning with continuous or discrete action selection.

---

## **`ActionSpaceMessage`**  

The `ActionSpaceMessage` structure defines the format for action transmission.  

| Field      | Type              | Description                                |
| ---------- | ----------------- | ------------------------------------------ |
| `action`   | `repeated int32`  | List of player actions.                    |
| `commands` | `repeated string` | List of Minecraft commands to be executed. |

---

## **Executing Minecraft Commands in a Gymnasium Wrapper**  

To execute Minecraft commands inside a Gymnasium environment, use the `add_commands` wrapper function:  

```python
self.get_wrapper_attr("add_commands")(
    [
        "setblock 1 2 3 minecraft:cake"
    ]
)
```

This example places a cake block at coordinates `(1,2,3)`, and the command is issued at the next `step()` call. You can pass multiple commands in the list to execute multiple actions in sequence.  