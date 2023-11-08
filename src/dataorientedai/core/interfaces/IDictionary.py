import abc


class IDictionary(abc.ABC):
    """В качестве ключа - зависимость, в качестве значения - стратегия(по
    входным параметрам возвратит ссылку на нужный объект)"""

    @abc.abstractmethod
    def __getitem__(self, key: str):
        pass

    @abc.abstractmethod
    def __setitem__(self, key: str, value: callable):
        pass

    @abc.abstractmethod
    def __contains__(self, key: str):
        pass

    @abc.abstractmethod
    def __delitem__(sefl, key: str):
        pass
