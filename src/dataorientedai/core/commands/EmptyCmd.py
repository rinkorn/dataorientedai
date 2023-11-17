from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IPlugin import IPlugin
from dataorientedai.core.IoC import IoC


class EmptyCmd(ICommand):
    def execute(self):
        pass


class InitEmptyCmdPlugin(IPlugin):
    def execute(self):
        IoC.resolve(
            "IoC.register",
            "Commands.EmptyCmd",
            lambda *args: EmptyCmd(*args),
        ).execute()
