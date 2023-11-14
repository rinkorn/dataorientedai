from dataorientedai.core.interfaces.ICommand import ICommand


class RepeateCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()
