from collections import deque
from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from get_device import get_device
from mineclip import MineCLIP
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class MineCLIPRewardWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, command: str, ckpt_path, **kwargs):
        self.env = env
        device = get_device()
        self.model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            hidden_dim=512,
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            pool_type="attn.d2.nh8.glusw",
            resolution=(160, 256),
        ).to(device)
        self.model.load_ckpt(ckpt_path, strict=True)
        self.command = command
        self.recent_frames = deque(maxlen=16)
        self.command_feat = self.model.encode_text([self.command])
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        self.recent_frames.append(info_obs["video"])
        video_feats = self.model.encode_video(self.recent_frames)
        reward, _ = self.model(
            video_feats, text_tokens=self.command_feat, is_video_features=True
        )
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options, fast_reset=fast_reset)
        info_obs = info["obs"]
        self.recent_frames.clear()
        self.recent_frames.append(info_obs["video"])
        return obs, info
