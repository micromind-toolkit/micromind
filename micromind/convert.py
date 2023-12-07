"""
Conversion from pytorch to different standard formats
for inference (ONNX, OpenVINO, tflite).

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
"""
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from .utils.helpers import get_logger
import micromind as mm

logger = get_logger()


@torch.no_grad()
def convert_to_onnx(
    net: Union[nn.Module, mm.MicroMind],
    save_path: Union[Path, str] = "model.onnx",
    simplify: bool = False,
):
    """Converts nn.Module to onnx and saves it to save_path.
    Optionally simplifies it. This function is internally used from `mm.MicroMind`.

    Arguments
    ---------
    net : Union[nn.Module, mm.MicroMind]
        PyTorch module to be exported.
    save_path : Union[Path, str]
        Output path for the ONNX model.
    simplify : bool
        `True` if you want to simplify the model. Defaults to False.

    Returns
    -------
    The path of the ONNX model. : Path
    """
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    x = torch.zeros([1] + list(net.input_shape))

    torch.onnx.export(
        net.cpu(),
        x,
        save_path,
        verbose=False,
        input_names=["input", "labels"],
        output_names=["output"],
        opset_version=11,
    )

    if simplify:
        import onnx
        import onnxsim

        onnx_model = onnx.load(save_path)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, save_path)

    logger.info(f"Saved converted ONNX model to {save_path}.")

    return save_path


@torch.no_grad()
def convert_to_openvino(net: Union[nn.Module, mm.MicroMind], save_path: Path) -> str:
    """Converts model to OpenVINO. Uses ONNX in the process and converts networks
    from channel-first to channel-last (for optimized inference).

    Arguments
    ---------
    net : nn.Module
        PyTorch module to be exported.
    save_path : Union[Path, str]
        Output path for the OpenVINO model.

    Returns
    -------
    The path of the XML model. : str

    """
    try:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import sys
        from pathlib import Path

        import onnx
        from loguru import logger
        from onnx_tf.backend import prepare
        from openvino.tools.mo import main as mo_main

    except Exception as e:
        print(str(e))
        print("Did you install micromind with conversion capabilities?")
        print("Please try again after pip install micromind[conversion].")
        exit(0)
    os.makedirs(save_path, exist_ok=True)
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    onnx_path = save_path.joinpath("model.onnx")
    onnx_model = onnx.load(convert_to_onnx(net, onnx_path, simplify=True))

    tf_rep = prepare(onnx_model)

    # Get the input tensor shape
    input_tensor = tf_rep.signatures[tf_rep.inputs[0]]
    input_shape = input_tensor.shape
    input_shape_str = "[" + ",".join([str(x) for x in input_shape]) + "]"

    cmd = [
        sys.executable,
        mo_main.__file__,
        "--input_model",
        str(onnx_path),
        "--input_shape",
        input_shape_str,
        "--output_dir",
        str(save_path),
        "--data_type",
        "FP32",
        "--silent",
        "True",
    ]

    os.popen(" ".join(cmd)).read()

    logger.info(f"Saved converted OpenVINO model to {save_path}.")

    return str(save_path.joinpath("model.xml"))


@torch.no_grad()
def convert_to_tflite(
    net: Union[nn.Module, mm.MicroMind],
    save_path: Union[Path, str],
    batch_quant: torch.Tensor = None,
) -> None:
    """Converts nn.Module to tf_lite, optionally quantizes it.

    Arguments
    ---------
    net : nn.Module
        PyTorch module to be exported.
    save_path : Union[Path, str]
        Output path for the OpenVINO model.
    batch_quant : torch.Tensor
        Optional batch for quantization. When passed, it is used to create the
        statistics of the quantized activations.

    """
    try:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import shutil
        import sys
        from pathlib import Path

        import numpy as np
        import tensorflow as tf
        from loguru import logger

    except Exception as e:
        print(str(e))
        print("Did you install micromind with conversion capabilities?")
        print("Please try again after pip install micromind[conversion].")
        exit(0)
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    if not (batch_quant is None):
        batch_quant = batch_quant.cpu()

    vino_sub = save_path.joinpath("vino")
    os.makedirs(vino_sub, exist_ok=True)
    vino_path = convert_to_openvino(net, vino_sub)
    if os.name == "nt":
        openvino2tensorflow_exe_cmd = [
            sys.executable,
            os.path.join(
                os.path.dirname(sys.executable), "Scripts", "openvino2tensorflow"
            ),
        ]
    else:
        openvino2tensorflow_exe_cmd = ["openvino2tensorflow"]

    cmd = openvino2tensorflow_exe_cmd + [
        "--model_path",
        str(vino_path),
        "--model_output_path",
        str(save_path),
        "--output_saved_model",
        "--output_no_quant_float32_tflite",
        "--non_verbose",
    ]

    os.popen(" ".join(cmd)).read()

    shutil.rmtree(vino_sub)

    if not (batch_quant is None):
        converter = tf.lite.TFLiteConverter.from_saved_model(str(save_path))

        def representative_dataset():
            for i, sample in enumerate(batch_quant):
                yield [np.expand_dims(sample.cpu().numpy(), axis=0)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.representative_dataset = representative_dataset

        tflite_quant_model = converter.convert()

        with open(save_path.joinpath("model.int8.tflite"), "wb") as f:
            f.write(tflite_quant_model)

    logger.info(f"Saved converted TFLite model to {save_path}.")
