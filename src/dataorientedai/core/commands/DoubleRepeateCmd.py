from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IPlugin import IPlugin
from dataorientedai.core.IoC import IoC


class DoubleRepeateCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


class InitDoubleRepeateCmdPlugin(IPlugin):
    def execute(self):
        IoC.resolve(
            "IoC.register",
            "Commands.DoubleRepeateCmd",
            lambda *args: DoubleRepeateCmd(*args),
        ).execute()
