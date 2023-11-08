import abc


class IExceptionHandler(abc.ABC):
    @abc.abstractmethod
    def handle(self):
        pass
