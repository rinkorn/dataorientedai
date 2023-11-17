from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IPlugin import IPlugin
from dataorientedai.core.IoC import IoC


class BridgeCmd(ICommand):
    def __init__(self, cmd: ICommand):
        self.cmd = cmd

    def inject(self, cmd):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


class InitBridgeCmdPlugin(IPlugin):
    def execute(self):
        IoC.resolve(
            "IoC.register",
            "Commands.BridgeCmd",
            lambda *args: BridgeCmd(*args),
        ).execute()


if __name__ == "__main__":
    from dataorientedai.core.commands.EmptyCmd import EmptyCmd
    from dataorientedai.core.commands.LogPrintCmd import LogPrintCmd
    from dataorientedai.core.commands.MacroCmd import MacroCmd
    from dataorientedai.core.IoC import InitScopeBasedIoCImplementationCmd

    InitScopeBasedIoCImplementationCmd().execute()
    base_scope = IoC.resolve("Scopes.new", IoC.resolve("Scopes.root"))
    IoC.resolve("Scopes.current.set", base_scope).execute()

    IoC.resolve(
        "IoC.register",
        "BridgeCmd",
        lambda *args: BridgeCmd(*args),
    ).execute()

    # cmd = BridgeCmd(MacroCmd(*[...]))
    # cmd.inject(LogPrintCmd(EmptyCmd, ValueError))
    # cmd.execute()

    scope = IoC.resolve("Scopes.new", IoC.resolve("Scopes.root"))
    IoC.resolve("Scopes.current.set", scope).execute()

    # IoC.resolve("Scopes.current.set", base_scope)

    cmd = IoC.resolve("BridgeCmd", MacroCmd(*[...]))
    cmd.inject(LogPrintCmd(EmptyCmd, ValueError))
    cmd.execute()
