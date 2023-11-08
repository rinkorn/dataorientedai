import abc


class ICommand(abc.ABC):
    @abc.abstractmethod
    def execute(self):
        pass
