from dataorientedai.core.interfaces.IDictionary import IDictionary
from dataorientedai.core.interfaces.IScope import IScope


class _Scope(IScope):
    """
    dependencies - словарик, где в виде ключа - зависимость, а в виде значения -
    стратегия (по входным параметрам возвратит ссылку на нужный объект)
    Не поток устанавливается в scope, а scope устанавливается в потоке.
    """

    def __init__(self, dependencies: IDictionary, parent: IScope):
        self.dependencies = dependencies
        self.parent = parent

    def resolve(self, key: str, *args: any):
        if key in self.dependencies:
            strategy = self.dependencies[key]
            return strategy(*args)
        else:
            return self.parent.resolve(key, *args)
