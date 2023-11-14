from dataorientedai.core.interfaces.ICommand import ICommand


class BridgeCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def inject(self, cmd):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


if __name__ == "__main__":
    from dataorientedai.core.commands.EmptyCmd import EmptyCmd
    from dataorientedai.core.commands.MacroCmd import MacroCmd

    cmds = [...]
    cmd = BridgeCmd(MacroCmd(*cmds))
    cmd.inject(EmptyCmd())
