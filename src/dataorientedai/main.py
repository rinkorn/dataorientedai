from pathlib import Path
from queue import Queue

from dataorientedai.ai.data.MnistNumpyToImageConvertable import (
    InitMnistNumpyToImageConvertableObjectCmd,
    MnistNumpyToImageConvertableAdapter,
    MnistNumpyToImageConvertCmd,
)
from dataorientedai.ai.data.MnistUbyteToNumpyConvertable import (
    InitMnistUbyteToNumpyConvertableObjectCmd,
    MnistUbyteToNumpyConvertableAdapter,
    MnistUbyteToNumpyConvertCmd,
)
from dataorientedai.ai.data.Unzippable import (
    InitUnzippableObjectCmd,
    UnzipCmd,
    UnzippableAdapter,
)
from dataorientedai.ai.Predictable import (
    InitPredictableObjectCmd,
    PredictableAdapter,
    PredictCmd,
)
from dataorientedai.ai.Trainable import (
    InitTrainableObjectCmd,
    TrainableAdapter,
    TrainCmd,
)
from dataorientedai.core.commands.DoubleRepeateCmd import DoubleRepeateCmd
from dataorientedai.core.commands.EmptyCmd import EmptyCmd
from dataorientedai.core.commands.HardStopCmd import (
    HardStopCmd,
    HardStoppableAdapter,
)
from dataorientedai.core.commands.LogPrintCmd import LogPrintCmd
from dataorientedai.core.commands.RepeateCmd import RepeateCmd
from dataorientedai.core.commands.SearchPluginsCmd import (
    SearchablePluginsAdapter,
    SearchPluginsCmd,
)
from dataorientedai.core.ContextDictionary import ContextDictionary
from dataorientedai.core.Dictionary import Dictionary
from dataorientedai.core.exceptions.BaseAppException import BaseAppException
from dataorientedai.core.handlers.ExceptionHandler import ExceptionHandler
from dataorientedai.core.IoC import InitScopeBasedIoCImplementationCmd, IoC
from dataorientedai.core.Processor import (
    InitProcessorContextCmd,
    Processable,
    Processor,
)
from dataorientedai.core.UObject import UObject


def main():
    from dataorientedai.core.IoC import InitScopeBasedIoCImplementationPlugin
    from dataorientedai.core.UObject import UObject

    obj = UObject()
    obj["path"] = Path(
        "/home/rinkorn/space/prog/python/free/project-dataorientedai/src/dataorientedai/"
    )
    obj["plugins"] = dict()
    obj["plugins"][
        "InitScopeBasedIoCImplementationPlugin"
    ] = InitScopeBasedIoCImplementationPlugin
    SearchPluginsCmd(SearchablePluginsAdapter(obj)).execute()


if __name__ == "__main__":
    # # %% First iteration
    # if __name__ == "__main__":
    #     obj = UObject()
    #     cmd = InitUnzippableObjectCmd(obj)
    #     cmd.execute()
    #     cmd = UnzipCmd(UnzippableAdapter(obj))
    #     cmd.execute()

    #     obj = UObject()
    #     cmd = InitMnistUbyteToNumpyConvertableObjecCmdt(obj)
    #     cmd.execute()
    #     convertable_obj = MnistUbyteToNumpyConvertableAdapter(obj)
    #     cmd = MnistUbyteToNumpyConvertCmd(convertable_obj)
    #     cmd.execute()

    #     obj = UObject()
    #     cmd = InitMnistNumpyToImageConvertableObjectCmd(obj)
    #     cmd.execute()
    #     cmd = MnistNumpyToImageConvertCmd(MnistNumpyToImageConvertableAdapter(obj))
    #     cmd.execute()

    #     obj = UObject()
    #     cmd = InitTrainableObjectCmd(obj)
    #     cmd.execute()
    #     cmd = TrainCmd(TrainableAdapter(obj))
    #     cmd.execute()

    #     obj = UObject()
    #     cmd = InitPredictableObjectCmd(obj)
    #     cmd.execute()
    #     cmd = PredictCmd(PredictableAdapter(obj))
    #     cmd.execute()

    # %% Third Iteration

    InitScopeBasedIoCImplementationCmd().execute()

    def func():
        print(f"root scope: {id(IoC.resolve('Scopes.root'))}")

        processor_context = Dictionary()
        InitProcessorContextCmd(processor_context).execute()
        processor = Processor(Processable(processor_context))

        scope = IoC.resolve("Scopes.new", IoC.resolve("Scopes.root"))
        IoC.resolve("Scopes.current.set", scope).execute()
        print(f"current scope: {id(IoC.resolve('Scopes.current'))}")
        # processor.wait()

        obj1 = UObject()
        obj2 = UObject()
        obj3 = UObject()
        obj4 = UObject()
        obj5 = UObject()
        queue = processor_context["queue"]
        queue.put(InitUnzippableObjectCmd(obj1))
        queue.put(InitMnistUbyteToNumpyConvertableObjectCmd(obj2))
        queue.put(InitMnistNumpyToImageConvertableObjectCmd(obj3))
        queue.put(InitTrainableObjectCmd(obj4))
        queue.put(InitPredictableObjectCmd(obj5))
        queue.put(UnzipCmd(UnzippableAdapter(obj1)))
        queue.put(
            MnistUbyteToNumpyConvertCmd(MnistUbyteToNumpyConvertableAdapter(obj2))
        )
        queue.put(
            MnistNumpyToImageConvertCmd(MnistNumpyToImageConvertableAdapter(obj3))
        )
        queue.put(TrainCmd(TrainableAdapter(obj4)))
        queue.put(PredictCmd(PredictableAdapter(obj5)))
        # queue.put(HardStopCmd(HardStoppableAdapter(processor_context)))
        return processor

    processor1 = func()
    processor2 = func()
    processor1.wait()
    processor2.wait()
