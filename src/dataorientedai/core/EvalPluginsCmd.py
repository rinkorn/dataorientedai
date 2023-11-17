import abc
from pathlib import Path

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IPlugin import IPlugin
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.IoC import IoC


# %%
class IEvaluablePlugins(abc.ABC):
    @abc.abstractclassmethod
    def get_plugins(self):
        pass


class EvaluablePluginsAdapter(abc.ABC):
    def __init__(self, o: IUObject):
        self.o = o

    def get_plugins(self):
        return self.o.__getitem__("plugins")

    def set_plugins(self, plugins):
        self.o.__setitem__("plugins", plugins)


class EvalPluginsCmd(ICommand):
    def __init__(self, o: IEvaluablePlugins):
        self.o = o

    def execute(self):
        plugins = self.o.get_plugins()
        # for plugin in plugins:
        #     plugin().execute()
        for i in range(len(plugins)):
            plugin = plugins.pop(0)
            plugin().execute()


class InitEvalPluginsCmdPlugin(IPlugin):
    def execute(self):
        IoC.resolve(
            "IoC.register",
            "Commands.EvalPluginsCmd",
            lambda *args: EvalPluginsCmd.load(args[0]),
        ).execute()


if __name__ == "__main__":
    from dataorientedai.core.Adapter import InitAdapterPlugin
    from dataorientedai.core.commands.SearchPluginsCmd import (
        SearchablePluginsAdapter,
        SearchPluginsCmd,
    )
    from dataorientedai.core.IoC import InitScopeBasedIoCImplementationPlugin
    from dataorientedai.core.UObject import UObject

    # InitScopeBasedIoCImplementationPlugin().execute()
    # root_scope = IoC.resolve("Scopes.root")
    # IoC.resolve("Scopes.current.set", root_scope).execute()
    # InitAdapterPlugin().execute()

    obj = UObject()
    obj["path"] = Path(
        "/home/rinkorn/space/prog/python/free/project-dataorientedai/src/dataorientedai/"
    )
    obj["plugins"] = [
        InitScopeBasedIoCImplementationPlugin,
        InitAdapterPlugin,
    ]
    SearchPluginsCmd(SearchablePluginsAdapter(obj)).execute()
    EvalPluginsCmd(EvaluablePluginsAdapter(obj)).execute()

    root_scope = IoC.resolve("Scopes.root")
    for key, value in root_scope.dependencies._store.items():
        print(key, "::", value)
