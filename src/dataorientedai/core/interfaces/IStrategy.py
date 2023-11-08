import abc


class IStrategy(abc.ABC):
    @abc.abstractstaticmethod
    def resolve(self, key: str, *args: any):
        pass
