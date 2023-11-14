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
from dataorientedai.core.commands.HardStopCmd import HardStopCmd
from dataorientedai.core.commands.LogPrintCmd import LogPrintCmd
from dataorientedai.core.commands.RepeateCmd import RepeateCmd
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


# # %% Second Iteration
# if __name__ == "__main__":
#     queue = Queue()
#     handler = ExceptionHandler()
#     handler.setup(
#         UnzipCmd,
#         BaseAppException,
#         lambda cmd, exc: queue.put(LogPrintCmd(cmd, exc)),
#     )
#     handler.setup(
#         MnistUbyteToNumpyConvertCmd,
#         BaseAppException,
#         lambda cmd, exc: queue.put(LogPrintCmd(cmd, exc)),
#     )
#     handler.setup(
#         MnistNumpyToImageConvertCmd,
#         BaseAppException,
#         lambda cmd, exc: queue.put(LogPrintCmd(cmd, exc)),
#     )
#     handler.setup(
#         TrainCmd,
#         BaseAppException,
#         lambda cmd, exc: queue.put(LogPrintCmd(cmd, exc)),
#     )
#     handler.setup(
#         PredictCmd,
#         BaseAppException,
#         lambda cmd, exc: queue.put(LogPrintCmd(cmd, exc)),
#     )
#     handler.setup(
#         RepeateCmd,
#         BaseAppException,
#         lambda cmd, exc: queue.put(LogPrintCmd(cmd, exc)),
#     )

#     obj = UObject()
#     queue.put(InitUnzippableObjectCmd(obj))
#     queue.put(UnzipCmd(UnzippableAdapter(obj)))

#     obj = UObject()
#     queue.put(InitMnistUbyteToNumpyConvertableObjectCmd(obj))
#     queue.put(MnistUbyteToNumpyConvertCmd(MnistUbyteToNumpyConvertableAdapter(obj)))

#     obj = UObject()
#     queue.put(InitMnistNumpyToImageConvertableObjectCmd(obj))
#     queue.put(MnistNumpyToImageConvertCmd(MnistNumpyToImageConvertableAdapter(obj)))

#     obj = UObject()
#     queue.put(InitTrainableObjectCmd(obj))
#     queue.put(TrainCmd(TrainableAdapter(obj)))

#     obj = UObject()
#     queue.put(InitPredictableObjectCmd(obj))
#     queue.put(PredictCmd(PredictableAdapter(obj)))

#     while True and not queue.empty():
#         cmd = queue.get()
#         try:
#             cmd.execute()
#         except Exception as e:
#             exc = type(e)
#             handler.handle(cmd, exc)
#             # IoC.resolve("ExceptionHandler", cmd, exc).execute()


# %% Third Iteration
if __name__ == "__main__":
    InitScopeBasedIoCImplementationCmd().execute()
    scope = IoC.resolve("scopes.new", IoC.resolve("scopes.root"))
    IoC.resolve("scopes.current.set", scope).execute()

    processor_context = Dictionary()
    InitProcessorContextCmd(processor_context).execute()
    queue = processor_context["queue"]

    obj = UObject()
    queue.put(InitUnzippableObjectCmd(obj))
    queue.put(UnzipCmd(UnzippableAdapter(obj)))

    obj = UObject()
    queue.put(InitMnistUbyteToNumpyConvertableObjectCmd(obj))
    queue.put(MnistUbyteToNumpyConvertCmd(MnistUbyteToNumpyConvertableAdapter(obj)))

    obj = UObject()
    queue.put(InitMnistNumpyToImageConvertableObjectCmd(obj))
    queue.put(MnistNumpyToImageConvertCmd(MnistNumpyToImageConvertableAdapter(obj)))

    obj = UObject()
    queue.put(InitTrainableObjectCmd(obj))
    queue.put(TrainCmd(TrainableAdapter(obj)))

    obj = UObject()
    queue.put(InitPredictableObjectCmd(obj))
    queue.put(PredictCmd(PredictableAdapter(obj)))

    queue.put(HardStopCmd(processor_context))
    # action
    processor = Processor(Processable(processor_context))
    # processor.wait()
    # # assert
    # assert queue.qsize() == 1
