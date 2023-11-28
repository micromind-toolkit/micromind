"""
Check if nn.Modules are convertible into quantized tflite networks.

Authors:
    - Francesco Paissan, 2023
"""
import torch


def test_onnx_phinet():
    from micromind.convert import convert_to_onnx
    from micromind.networks import PhiNet

    save_path = "temp.onnx"

    in_shape = (3, 224, 224)
    net = PhiNet(in_shape, compatibility=False, include_top=True)

    convert_to_onnx(net, save_path, simplify=True)
    import os

    os.remove(save_path)

    convert_to_onnx(net, save_path, simplify=False)

    import os

    os.remove(save_path)


def test_onnx_xinet():
    from micromind.convert import convert_to_onnx
    from micromind.networks import XiNet

    save_path = "temp.onnx"

    in_shape = (3, 224, 224)
    net = XiNet(in_shape, include_top=True)

    convert_to_onnx(net, save_path, simplify=True)
    import os

    os.remove(save_path)

    convert_to_onnx(net, save_path, simplify=False)

    import os

    os.remove(save_path)


def test_openvino_phinet():
    from micromind.convert import convert_to_openvino
    from micromind.networks import PhiNet

    save_dir = "vino"

    in_shape = (3, 224, 224)
    net = PhiNet(in_shape, compatibility=False, include_top=True)

    convert_to_openvino(net, save_dir)

    import shutil

    shutil.rmtree(save_dir)


def test_openvino_xinet():
    from micromind.convert import convert_to_openvino
    from micromind.networks import XiNet

    save_dir = "vino"

    in_shape = (3, 224, 224)
    net = XiNet(in_shape)

    convert_to_openvino(net, save_dir)

    import shutil

    shutil.rmtree(save_dir)


def test_tflite_phinet():
    from micromind.convert import convert_to_tflite
    from micromind.networks import PhiNet

    save_path = "tflite"

    in_shape = (3, 224, 224)
    net = PhiNet(in_shape, compatibility=False, include_top=True)

    convert_to_tflite(net, save_path)

    import shutil

    shutil.rmtree(save_path)

    temp = torch.Tensor(100, in_shape[1], in_shape[2], in_shape[0])
    convert_to_tflite(net, save_path, temp)

    import shutil

    shutil.rmtree(save_path)


def test_tflite_xinet():
    from micromind.convert import convert_to_tflite
    from micromind.networks import XiNet

    save_path = "tflite"

    in_shape = (3, 224, 224)
    net = XiNet(in_shape)

    convert_to_tflite(net, save_path)

    import shutil

    shutil.rmtree(save_path)

    temp = torch.Tensor(100, in_shape[1], in_shape[2], in_shape[0])
    convert_to_tflite(net, save_path, temp)

    import shutil

    shutil.rmtree(save_path)
