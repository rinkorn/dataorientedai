import abc


class IUObject(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, key: str):
        pass

    @abc.abstractmethod
    def __setitem__(self, key: str, value: any):
        pass
