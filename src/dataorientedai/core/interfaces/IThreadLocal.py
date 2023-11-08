import abc


class IThreadLocal:
    @abc.abstractmethod
    def __getitem__(self):
        pass

    @abc.abstractmethod
    def __setitem__(self):
        pass

    @abc.abstractmethod
    def __contains__(self):
        pass
