# Dueling dqns for sound, vision, and bimodal inputs
from typing import Optional

from code.wrappers.action import ActionWrapper
from code.models.dueling_dqn_base import DuelingDQNAgentBase


class ManualVisionAgent(DuelingDQNAgentBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        device,
        stack_size=None,
    ):
        self.action_dim = action_dim
        self.enabled_actions = [
            ActionWrapper.JUMP,
            ActionWrapper.NO_OP,
            ActionWrapper.JUMP_USE,
            ActionWrapper.TURN_RIGHT,
            ActionWrapper.TURN_LEFT,
            ActionWrapper.LOOK_UP,
            ActionWrapper.LOOK_DOWN,
        ]
        self.actions = [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            2,
            2,
            2,
            0,
            1,
            1,
            0,
        ]
        self.read_idx = 0

    @property
    def config(self):
        return {
            "architecture": "Manual",
            "state_dim": 0,
            "action_dim": None,
            "kernel_size": 0,
            "stride": 0,
            "hidden_dim": 0,
            "buffer_size": 0,
            "learning_rate": 0,
            "gamma": None,
            "batch_size": None,
            "optimizer": None,
            "loss_fn": None,
            "weight_decay": 0,
        }

    def select_action(self, state, testing, **kwargs):
        # print("Selecting action")
        epsilon = kwargs["epsilon"]
        # if np.random.rand() <= epsilon and not testing:
        #     # print("random action")
        #     return np.random.choice(self.action_dim)
        # else:
        if self.read_idx > len(self.actions) - 1:
            self.read_idx = 0
            # print(f"Action: {self.actions[self.read_idx]}")
            return self.actions[self.read_idx]
        else:
            action = self.actions[self.read_idx]
            self.read_idx += 1
            # print(f"Action: {action}")
            return action

    def update_model(self) -> Optional[float]:
        return 0

    def update_target_model(self):
        pass

    def add_experience(self, state, action, next_state, reward, done):
        pass
