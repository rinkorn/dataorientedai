import abc


class IQueue(abc.ABC):
    @abc.abstractmethod
    def put(self, item):
        pass

    @abc.abstractmethod
    def get(self):
        pass

    @abc.abstractmethod
    def empty(self):
        pass
