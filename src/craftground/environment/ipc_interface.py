from abc import ABC, abstractmethod

from proto.action_space_pb2 import ActionSpaceMessageV2
from proto.initial_environment_pb2 import InitialEnvironmentMessage
from proto.observation_space_pb2 import ObservationSpaceMessage


class IPCInterface(ABC):
    @abstractmethod
    def send_action(self, message: ActionSpaceMessageV2):
        pass

    @abstractmethod
    def read_observation(self) -> ObservationSpaceMessage:
        pass

    @abstractmethod
    def send_initial_environment(self, message: InitialEnvironmentMessage):
        pass
