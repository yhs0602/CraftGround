import abc


class Agent(abc.ABC):
    @property
    @abc.abstractmethod
    def config(self):
        pass

    @abc.abstractmethod
    def select_action(self, obs, testing, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    def update_model(self):
        pass
