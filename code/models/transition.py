from collections import namedtuple
from typing import TypeAlias, List

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)

BimodalTransition = namedtuple(
    "BiModalTransition",
    ("audio", "video", "action", "next_audio", "next_video", "reward", "done"),
)

BimodalTokenTransition = namedtuple(
    "BiModalTokenTransition",
    (
        "audio",
        "video",
        "token",
        "action",
        "next_audio",
        "next_video",
        "next_token",
        "reward",
        "done",
    ),
)

Episode: TypeAlias = List[Transition]
BimodalEpisode: TypeAlias = List[BimodalTransition]
