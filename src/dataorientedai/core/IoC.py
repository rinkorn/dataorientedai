import abc

from dataorientedai.core.ConcurrentDictionary import (
    ConcurrentDictionary,
    NotThreadSafeDictionary,
)
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IDictionary import IDictionary
from dataorientedai.core.interfaces.IIoC import IIoC
from dataorientedai.core.interfaces.IScope import IScope
from dataorientedai.core.interfaces.IStrategy import IStrategy
from dataorientedai.core.LeafScope import LeafScope
from dataorientedai.core.Scope import Scope
from dataorientedai.core.ThreadLocal import ThreadLocal
from dataorientedai.core.UObject import UObject


# %%
class _SetupStrategyCmd(ICommand):
    def __init__(self, new_strategy):
        self.new_strategy = new_strategy

    def execute(self):
        IoC._strategy = self.new_strategy


class IoC(IIoC):
    @staticmethod
    def resolve(key: str, *args: any):
        return IoC._strategy(key, *args)

    @staticmethod
    def _default_strategy(key: str, *args: any):
        """Если нужно будет заменить именно дефолтную стратегию (то есть убрать
        возможность делать "IoC.setup_strategy", то вернём ссылку
        на IoC._default_strategy, чтобы где-то как-то её подменить или удалить.
        """
        if key == "IoC.setup_strategy":
            strategy = args[0]
            return _SetupStrategyCmd(strategy)
        elif key == "IoC.default_strategy":
            return IoC._default_strategy
        else:
            raise ValueError(
                f"Unknown IoC dependency key {key}. "
                f"Make sure that {key} has been registered"
            )

    _strategy: callable = _default_strategy


# %%
class ScopeBasedResolveDependencyStrategy(IStrategy):
    _root: Scope = None
    _current_scopes = ThreadLocal()

    @staticmethod
    def _default_scope():
        return ScopeBasedResolveDependencyStrategy._root

    @staticmethod
    def resolve(key: str, *args: any):
        # if key == "IoC.setup_strategy":
        #     return SetupStrategyCmd(args[0])
        if key == "Scopes.root":
            return ScopeBasedResolveDependencyStrategy._root
        else:
            # Не поток устанавливается в scope, а scope устанавливается в потоке.
            scope = ScopeBasedResolveDependencyStrategy._current_scopes["value"]
            if scope is None:
                scope = ScopeBasedResolveDependencyStrategy._default_scope()
            return scope.resolve(key, *args)


# %%
class InitSingleThreadScopeCmd(ICommand):
    def execute(self) -> None:
        IoC.resolve(
            "IoC.register",
            "Scopes.storage",
            lambda *args: NotThreadSafeDictionary(),
        ).execute()


def RegisterIoCDependencyException(Exception):
    pass


def UnregisterIoCDependencyException(Exception):
    pass


class _RegisterIoCDependencyCmd(ICommand):
    def __init__(self, key: str, strategy: callable):
        self.key = key
        self.strategy = strategy

    def execute(self):
        try:
            current_scope = ScopeBasedResolveDependencyStrategy._current_scopes["value"]
            current_scope.dependencies.__setitem__(
                self.key,
                self.strategy,
            )
        except BaseException as e:
            raise RegisterIoCDependencyException("Can't register dependency")


class _UnregisterIoCDependencyCmd(ICommand):
    def __init__(self, key: str):
        self.key = key

    def execute(self):
        try:
            current_scope = ScopeBasedResolveDependencyStrategy._current_scopes["value"]
            current_scope.dependencies.__delitem__(self.key)
        except BaseException as e:
            raise UnregisterIoCDependencyException("Can't unregister dependency")


class _SetScopeInCurrentThreadCmd(ICommand):
    def __init__(self, scope):
        self.scope = scope

    def execute(self):
        ScopeBasedResolveDependencyStrategy._current_scopes.__setitem__(
            "value",
            self.scope,
        )


class InitScopeBasedIoCImplementationCmd(ICommand):
    def execute(self):
        if ScopeBasedResolveDependencyStrategy._root is not None:
            return

        dependencies = ConcurrentDictionary()

        # scopes.storage - словарик для всех scopes, которые есть в приложении
        dependencies.__setitem__(
            "Scopes.storage",
            lambda *args: ConcurrentDictionary(),
        )

        # scopes.new - команда, которая создаёт storage когда это необходимо
        dependencies.__setitem__(
            "Scopes.new",
            lambda *args: Scope(
                IoC.resolve("Scopes.storage"),
                args[0],
            ),
        )

        # scopes.current - получить доступ к текущему scope
        current_scope = ScopeBasedResolveDependencyStrategy._current_scopes["value"]
        default_scope = ScopeBasedResolveDependencyStrategy._default_scope
        dependencies.__setitem__(
            "Scopes.current",
            current_scope if current_scope is not None else default_scope,
        )

        # scopes.current - устанвоить scope в текущем потоке
        dependencies.__setitem__(
            "Scopes.current.set",
            lambda *args: _SetScopeInCurrentThreadCmd(args[0]),
        )

        dependencies.__setitem__(
            "IoC.register",
            lambda *args: _RegisterIoCDependencyCmd(args[0], args[1]),
        )

        dependencies.__setitem__(
            "IoC.unregister",
            lambda *args: _UnregisterIoCDependencyCmd(args[0]),
        )

        root_scope = Scope(
            dependencies,
            # LeafScope(IoC.resolve("IoC.default_strategy")),
            parent=None,
        )

        ScopeBasedResolveDependencyStrategy._root = root_scope

        IoC.resolve(
            "IoC.setup_strategy",
            ScopeBasedResolveDependencyStrategy.resolve,
        ).execute()

        _SetScopeInCurrentThreadCmd(root_scope).execute()


if __name__ == "__main__":
    # InitSingleThreadScopeCmd().execute()
    InitScopeBasedIoCImplementationCmd().execute()

    IoC.resolve("Scopes.current.set", IoC.resolve("Scopes.root"))
    IoC.resolve(
        "IoC.register",
        "UObject",
        lambda *args: UObject(),
    ).execute()

    scope = IoC.resolve("Scopes.new", IoC.resolve("Scopes.root"))
    IoC.resolve("Scopes.current.set", scope)

    IoC.resolve(
        "IoC.register",
        "a",
        lambda *args: f"just a! and params: {args}",
    ).execute()

    print(IoC.resolve("a", 123, 456))

    IoC.resolve(
        "IoC.unregister",
        "a",
    ).execute()

    from dataorientedai.core.commands.BridgeCmd import BridgeCmd
    from dataorientedai.core.commands.EmptyCmd import EmptyCmd
    from dataorientedai.core.commands.LogPrintCmd import LogPrintCmd
    from dataorientedai.core.commands.MacroCmd import MacroCmd

    IoC.resolve(
        "Scopes.current.set",
        IoC.resolve("Scopes.root"),
    ).execute()
    IoC.resolve(
        "IoC.register",
        "BridgeCmd",
        lambda *args: BridgeCmd(*args),
    ).execute()
    IoC.resolve(
        "Scopes.current.set",
        scope,
    ).execute()
    cmd = IoC.resolve("BridgeCmd", MacroCmd(*[...]))
    cmd.inject(LogPrintCmd(EmptyCmd, ValueError))
    cmd.execute()
