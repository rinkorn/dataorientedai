from dataorientedai.core.interfaces.IScope import IScope


class LeafScope(IScope):
    def __init__(self, strategy: callable):
        self._strategy = strategy

    def resolve(self, key: str, *args: any):
        return self._strategy(key, *args)
