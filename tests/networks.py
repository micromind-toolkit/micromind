"""
Check for scriptability and ONNX exportability with operation set 11.

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
"""
import torch
import torch.nn as nn

def test_phinet():
    from phinet import PhiNet

    with torch.no_grad():
        in_shape = list((3, 224, 224))
        x = torch.zeros([1] + in_shape)
        net = PhiNet(in_shape, compatibility=False)
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

    import os
    os.remove("temp.onnx")

def test_tflite():
    from onnx_tf.backend import prepare
    import onnx
    import tensorflow as tf
    import numpy as np
    import onnxsim
    from phinet import PhiNet

    with torch.no_grad():
        in_shape = list((3, 224, 224))
        x = torch.zeros([1] + in_shape)
        net = PhiNet(in_shape, compatibility=True)
        net = net.to(memory_format=torch.channels_last)
        net.eval()
        torch.onnx.export(
            net,
            x.to(memory_format=torch.channels_last),
            "temp.onnx",
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            export_params=True
        )

        tf_model_path = 'model_tf'
        tflite_model_path = 'model_tflite.tflite'
    
        onnx_model = onnx.load("temp.onnx")
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "Model simplification failed"


        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_model_path) 
    
    
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        def representative_dataset_gen():
            for _ in range(10):
                temp = np.random.randn(*in_shape)
                temp = np.expand_dims(temp, 0).astype(np.float32)
                yield [temp]
    
        converter.representative_dataset = representative_dataset_gen   
        tflite_model = converter.convert()
    
        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

    import os
    import shutil
    shutil.rmtree(tf_model_path)
    os.remove("temp.onnx")
    os.remove(tflite_model_path)
      
