import abc


class IIoC(abc.ABC):
    @abc.abstractstaticmethod
    def resolve(key: str, *args: any):
        pass

    @abc.abstractstaticmethod
    def _default_strategy(key: str, *args: any):
        pass

    @abc.abstractproperty
    def _strategy():
        pass
