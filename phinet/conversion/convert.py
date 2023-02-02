"""
Conversion from pytorch to different standard formats for inference (ONNX, OpenVINO, tflite).

Authors:
	- Francesco Paissan, 2023
	- Alberto Ancilotto, 2023
"""
import torch
import torch.nn as nn

import onnx
import onnxsim
import numpy as np
import tensorflow as tf
from onnx_tf.backend import prepare
from openvino.tools.mo import main as mo_main

from pathlib import Path
import shutil
import sys
import os

@torch.no_grad()
def convert_to_onnx(net: nn.Module, save_path: Path, simplify: bool = True):
	""" Converts nn.Module to onnx and saves it to save_path. Optionally simplifies it."""
	x = torch.zeros([1] + net.input_shape)

	torch.onnx.export(
		net,
		x,
		save_path,
		verbose=False,
		input_names=["input"],
		output_names=["output"],
		opset_version=11,
	)

	if simplify:
		onnx_model = onnx.load(save_path)
		onnx_model, check = onnxsim.simplify(onnx_model)
		onnx.save(onnx_model, save_path)

	return onnx.load(save_path)

@torch.no_grad()
def convert_to_openvino(net: nn.Module, save_dir: Path) -> str:
	""" Converts nn.Module to OpenVINO. """
	os.makedirs(save_dir, exist_ok=True)
	if not isinstance(save_dir, Path):
		save_dir = Path(save_dir)

	onnx_path = save_dir.joinpath("model.onnx")
	onnx_model = convert_to_onnx(net, onnx_path, simplify=True)

	tf_rep = prepare(onnx_model)

	# Get the input tensor shape
	input_tensor = tf_rep.signatures[tf_rep.inputs[0]]
	input_shape = input_tensor.shape
	input_shape_str = '[' + ','.join([str(x) for x in input_shape]) + ']'
	
	cmd = [ 
		sys.executable, mo_main.__file__, 
		'--input_model', str(onnx_path),
		'--input_shape', input_shape_str,
		'--output_dir', str(save_dir),
		'--data_type', 'FP32'
	
	]

	os.system(" ".join(cmd))

	return str(save_dir.joinpath("model.xml"))

@torch.no_grad()
def convert_to_tflite(net: nn.Module, save_path: Path, batch_quant: torch.Tensor = None) -> None:
	""" Converts nn.Module to tf_lite, optionally quantizes it. """
	if not isinstance(save_path, Path):
		save_path = Path(save_path)

	vino_sub = save_path.joinpath("vino")
	os.makedirs(vino_sub, exist_ok=True)
	vino_path = convert_to_openvino(net, vino_sub)
	if os.name == 'nt':
		openvino2tensorflow_exe_cmd = [sys.executable, os.path.join(os.path.dirname(sys.executable), 'openvino2tensorflow')]
	else:
		openvino2tensorflow_exe_cmd = ['openvino2tensorflow']
		
	cmd = openvino2tensorflow_exe_cmd + [ 
		'--model_path', str(vino_path),
		'--model_output_path', str(save_path),
		'--output_saved_model',
		'--output_no_quant_float32_tflite'
	]

	os.system(" ".join(cmd))

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
		
		with open(save_path.joinpath("model.int8.tflite"), 'wb') as f:
			f.write(tflite_quant_model)	

	
	
