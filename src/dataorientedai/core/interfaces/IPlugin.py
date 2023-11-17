import abc


class IPlugin(abc.ABC):
    @abc.abstractmethod
    def execute(self):
        pass
