import abc


class IScope(abc.ABC):
    @abc.abstractmethod
    def resolve(key: str, *args: any):
        pass
