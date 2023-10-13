"""
Conversion from pytorch to different standard formats
for inference (ONNX, OpenVINO, tflite).

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
"""
from pathlib import Path
from loguru import logger
from typing import Union
import torch.nn as nn
import torch
import os


@torch.no_grad()
def convert_to_onnx(
    net: nn.Module,
    save_path: Union[Path, str] = "model.onnx",
    simplify: bool = False,
    replace_forward: bool = False,
):
    """Converts nn.Module to onnx and saves it to save_path.
    Optionally simplifies it."""
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    x = torch.zeros([1] + list(net.input_shape))

    if replace_forward:
        # add forward to ModuleDict
        bound_method = net.forward.__get__(net.modules, net.modules.__class__)
        setattr(net.modules, "forward", bound_method)

        net.modules.input_shape = net.input_shape
        net = net.modules
        x = [torch.zeros([1] + list(net.input_shape)), None]

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
def convert_to_openvino(
    net: nn.Module, save_path: Path, replace_forward: bool = False
) -> str:
    """Converts nn.Module to OpenVINO."""
    try:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import sys
        from pathlib import Path
        from loguru import logger

        import onnx
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
    onnx_model = onnx.load(
        convert_to_onnx(net, onnx_path, simplify=True, replace_forward=replace_forward)
    )

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
    net: nn.Module,
    save_path: Path,
    batch_quant: torch.Tensor = None,
    replace_forward: bool = False,
) -> None:
    """Converts nn.Module to tf_lite, optionally quantizes it."""
    try:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import shutil
        import sys
        from pathlib import Path
        from loguru import logger

        import numpy as np
        import tensorflow as tf

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
    vino_path = convert_to_openvino(net, vino_sub, replace_forward=replace_forward)
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
