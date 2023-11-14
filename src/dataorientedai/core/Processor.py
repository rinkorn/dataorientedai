import abc
import threading

from dataorientedai.core.BlockingCollection import BlockingCollection
from dataorientedai.core.interfaces.ICommand import ICommand
from dataorientedai.core.interfaces.IDictionary import IDictionary


class IProcessable(abc.ABC):
    @abc.abstractmethod
    def can_continue(self):
        pass

    @abc.abstractmethod
    def thread_timeout(self):
        pass

    @abc.abstractmethod
    def process(self):
        pass


class Processable(IProcessable):
    def __init__(self, context: IDictionary):
        self.o = context

    def can_continue(self):
        return self.o["can_continue"]

    def thread_timeout(self):
        return self.o["thread_timeout"]

    def process(self):
        self.o["process"].__call__()


class Processor:
    def __init__(self, context: IProcessable):
        self.context = context
        self.thread = threading.Thread(
            target=self.evaluation,
            daemon=True,
        )
        self.thread.start()

    def wait(self):
        self.thread.join(
            timeout=self.context.thread_timeout(),
        )

    def evaluation(self):
        while self.context.can_continue():
            self.context.process()


class InitProcessorContextCmd(ICommand):
    def __init__(self, context: IDictionary):
        self.o = context

    def execute(self):
        queue = BlockingCollection()

        def process():
            cmd = queue.get()
            try:
                cmd.execute()
                print(f"Executed: {cmd.__class__.__name__}")
            except Exception as e:
                exc = type(e)
                try:
                    # handler.handle(cmd, exc)
                    # ExceptionHandler.handle(cmd, exc).execute()
                    # IoC.resolve("ExceptionHandler", cmd, exc).execute()
                    print(f"Error! {exc} in {type(cmd)}")
                except Exception as e:
                    print(f"Fatal error! {exc} in {type(cmd)}")

        self.o["can_continue"] = True
        self.o["queue"] = queue
        self.o["process"] = process
        self.o["thread_timeout"] = 10


if __name__ == "__main__":
    from dataorientedai.core.commands.EmptyCmd import EmptyCmd
    from dataorientedai.core.commands.HardStopCmd import (
        HardStopCmd,
        HardStoppableAdapter,
    )
    from dataorientedai.core.commands.SoftStopCmd import (
        SoftStopCmd,
        SoftStoppableAdapter,
    )
    from dataorientedai.core.Dictionary import Dictionary

    # from dataorientedai.core.UObject import UObject

    # test_HardStop
    # assign
    obj = Dictionary()
    InitProcessorContextCmd(obj).execute()
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(HardStopCmd(HardStoppableAdapter(obj)))
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(EmptyCmd())
    # action
    processor = Processor(Processable(obj))
    processor.wait()
    # # assert
    assert obj["queue"].qsize() == 2
    print(obj["queue"].qsize())
    print()

    # test_SoftStop
    # assign
    obj = Dictionary()
    InitProcessorContextCmd(obj).execute()
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(SoftStopCmd(SoftStoppableAdapter(obj)))
    obj["queue"].put(EmptyCmd())
    obj["queue"].put(EmptyCmd())
    # action
    processor = Processor(Processable(obj))
    processor.wait()
    # assert
    assert obj["queue"].empty()
    print(obj["queue"].qsize())
    print()
