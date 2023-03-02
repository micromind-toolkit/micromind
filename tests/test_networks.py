"""
Check if nn.Modules are convertible into quantized tflite networks.

Authors:
    - Francesco Paissan, 2023
"""
import torch


def test_onnx():
    from micromind import PhiNet
    from micromind.conversion import convert_to_onnx

    save_path = "temp.onnx"

    in_shape = list((3, 224, 224))
    net = PhiNet(in_shape, compatibility=False)

    convert_to_onnx(net, save_path, simplify=True)
    import os

    os.remove(save_path)

    convert_to_onnx(net, save_path, simplify=False)

    import os

    os.remove(save_path)


def test_openvino():
    from micromind import PhiNet
    from micromind.conversion import convert_to_openvino

    save_dir = "vino"

    in_shape = list((3, 224, 224))
    net = PhiNet(in_shape, compatibility=False)

    convert_to_openvino(net, save_dir)

    import shutil

    shutil.rmtree(save_dir)


def test_tflite():
    from micromind import PhiNet
    from micromind.conversion import convert_to_tflite

    save_path = "tflite"

    in_shape = list((3, 224, 224))
    net = PhiNet(in_shape, compatibility=False)

    convert_to_tflite(net, save_path)

    import shutil

    shutil.rmtree(save_path)

    temp = torch.Tensor(100, in_shape[1], in_shape[2], in_shape[0])
    convert_to_tflite(net, save_path, temp)

    import shutil

    shutil.rmtree(save_path)
