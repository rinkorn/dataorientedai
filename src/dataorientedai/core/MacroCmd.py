from dataorientedai.core.exceptions.CommandException import CommandException
from dataorientedai.core.interfaces.ICommand import ICommand


class MacroCmd(ICommand):
    """Реализовать простейшую макрокоманду.
    Здесь простейшая - это значит, что при выбросе исключения
    вся последовательность команд приостанавливает свое выполнение,
    а макрокоманда выбрасывает CommandException.
    """

    def __init__(self, *cmds):
        self.cmds = cmds

    def execute(self):
        try:
            for cmd in self.cmds:
                cmd.execute()
        except Exception:
            raise CommandException("Can't execute MacroCmd")
