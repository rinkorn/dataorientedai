import abc
import importlib
import inspect

from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IPlugin import IPlugin
from dataorientedai.core.interfaces.IUObject import IUObject
from dataorientedai.core.IoC import IoC


class ISearchablePlugins(abc.ABC):
    @abc.abstractclassmethod
    def get_path(self):
        pass

    @abc.abstractclassmethod
    def get_plugins(self):
        pass

    @abc.abstractclassmethod
    def set_plugins(self):
        pass


class SearchablePluginsAdapter(abc.ABC):
    def __init__(self, o: IUObject):
        self.o = o

    def get_path(self):
        return self.o.__getitem__("path")

    def get_plugins(self):
        return self.o.__getitem__("plugins")

    def set_plugins(self, plugins):
        self.o.__setitem__("plugins", plugins)


class SearchPluginsCmd(ICommand):
    def __init__(self, o: ISearchablePlugins):
        self.o = o

    def execute(self):
        path = self.o.get_path()
        if not path.exists():
            return

        filenames = []
        for fn in path.rglob("*.py"):
            if fn.name == "__init__.py":
                continue
            filenames.append(fn)

        plugins = self.o.get_plugins()
        for fn in filenames:
            # module_path: str = str(fn.parent / fn.stem).replace("/", ".")
            module_path = str(fn.parent / fn.stem).split("/src/")[1].replace("/", ".")
            # module_path: str = fn.replace("/", ".")
            module = importlib.import_module(module_path)
            for obj_name, obj in inspect.getmembers(module):
                if not inspect.isclass(obj):
                    continue
                if not issubclass(obj, IPlugin):
                    continue
                if obj == IPlugin:
                    continue
                if obj not in plugins:
                    plugins.append(obj)
                # plugin_name = str(module_path) + "::" + obj_name
                # plugin = obj
                # if plugin_name not in plugins:
                #     plugins[plugin_name] = plugin
                # print(plugin_name)

        self.o.set_plugins(plugins)


class InitSearchPluginsCmdPlugin(IPlugin):
    def execute(self):
        IoC.resolve(
            "IoC.register",
            "Commands.SearchPluginsCmd",
            lambda *args: SearchPluginsCmd.load(args[0]),
        ).execute()


if __name__ == "__main__":
    from pathlib import Path

    from dataorientedai.core.IoC import InitScopeBasedIoCImplementationPlugin
    from dataorientedai.core.UObject import UObject

    obj = UObject()
    obj["path"] = Path(
        "/home/rinkorn/space/prog/python/free/project-dataorientedai/src/dataorientedai/"
    )
    obj["plugins"] = list()
    obj["plugins"].append(InitScopeBasedIoCImplementationPlugin)
    obj["plugins"].append(InitSearchPluginsCmdPlugin)
    SearchPluginsCmd(SearchablePluginsAdapter(obj)).execute()

    for plugin in obj["plugins"]:
        print(plugin.__name__)
