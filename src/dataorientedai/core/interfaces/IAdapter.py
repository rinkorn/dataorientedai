import abc


class IAdapter(abc.ABC):
    @abc.abstractstaticmethod
    def generate(dependency_space: str, IClass: object):
        pass
