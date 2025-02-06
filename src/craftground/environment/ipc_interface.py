from abc import ABC, abstractmethod

from ..proto.action_space_pb2 import ActionSpaceMessageV2
from ..proto.observation_space_pb2 import ObservationSpaceMessage


class IPCInterface(ABC):
    port: int

    @abstractmethod
    def send_action(self, message: ActionSpaceMessageV2):
        pass

    @abstractmethod
    def read_observation(self) -> ObservationSpaceMessage:
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        pass

    @abstractmethod
    def destroy(self):
        pass
