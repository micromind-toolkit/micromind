"""
Check for scriptability and ONNX exportability with operation set 11.

Authors:
    - Francesco Paissan, 2023
"""
import torch
import torch.nn as nn

def test_phinet():
    from phinet import PhiNet

    with torch.no_grad():
        in_shape = list((3, 224, 224))
        x = torch.zeros([1] + in_shape)
        net = PhiNet(in_shape, compatibility=True)
        net.eval()
        torch.onnx.export(
            net,
            x,
            "temp.onnx",
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
        )
