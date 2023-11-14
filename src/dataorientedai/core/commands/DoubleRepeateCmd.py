from dataorientedai.core.interfaces.ICommand import ICommand


class DoubleRepeateCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()
