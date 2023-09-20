from collections import deque
from typing import SupportsFloat, Any, Optional

import numpy as np
import torch
from gymnasium.core import WrapperActType, WrapperObsType

from get_device import get_device
from mineclip import MineCLIP
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class MineCLIPRewardWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, command: str, ckpt_path, **kwargs):
        self.env = env
        self.device = get_device()
        self.model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            hidden_dim=512,
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            pool_type="attn.d2.nh8.glusw",
            resolution=(160, 256),  # (160, 256),
        ).to(self.device)
        self.model.load_ckpt(ckpt_path, strict=True)
        self.model.eval()
        self.command = command
        self.recent_frames_features = deque(maxlen=16)
        self.command_feat = self.model.encode_text([self.command])
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = info["rgb"]
        with torch.no_grad():
            image_feat = self.model.forward_image_features(
                torch.from_numpy(rgb).to(self.device).unsqueeze(0)
            ).squeeze(0)
        self.recent_frames_features.append(image_feat)
        if len(self.recent_frames_features) > 4:
            combined_tensor = torch.stack(
                list(self.recent_frames_features), dim=0
            ).unsqueeze(0)
            # print(f"{combined_tensor.shape=}, {type(combined_tensor)=}")
            # requires 5D tensor
            with torch.no_grad():
                video_features = self.model.forward_video_features(combined_tensor)
                # print(f"{video_features.shape=} {combined_tensor.shape=}")
                reward, _ = self.model(
                    video_features,
                    text_tokens=self.command_feat,
                    is_video_features=True,
                )
                # print(f"{reward.shape=}")
            del combined_tensor
            del video_features
            reward = reward.cpu().numpy().item()
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        rgb = info["rgb"]
        self.recent_frames_features.clear()
        with torch.no_grad():
            image_feat = self.model.forward_image_features(
                torch.from_numpy(rgb).to(self.device).unsqueeze(0)
            ).squeeze(0)
        self.recent_frames_features.append(image_feat)
        return obs, info
