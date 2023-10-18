"""
Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

This code is used to obtain a version of the yolov8 model with pre-trained
weights that can be used with the implementation of the network present in
this repository.

The user must provide the version of the model he intends to use on the
command line (available options are "n", "s", "m", "l", "x").
The desired model is downloaded and each model parameter is copied into the
model state dictionary initialized via our yolov8 implementation.
The model state dictionary is finally saved in the pytorch .pt format.
"""

import torch
import sys
import requests
import os

from micromind.networks.modules import YOLOv8
from yolo_helpers import get_variant_multiples


def download_variant(variant):
    """
    This function downloads the weights of the desired pre-trained model
    from the official Ultralytics GitHub repository
    https://github.com/ultralytics/assets/releases/ and saves the
    downloaded file in the current working directory.

    Arguments
    ---------
    variant : str
        yolov8 model variant. Default to 'l'.

    Returns
    -------
        None
    """
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8{variant}.pt"
    file_name = f"yolov8{variant}.pt"
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"File {file_name} downloaded successfully.")
    else:
        print("Unable to download the file. Status code: ", response.status_code)


if __name__ == "__main__":

    valid_conf = ["n", "s", "m", "l", "x"]
    if len(sys.argv) < 2 or len(sys.argv[1]) > 1 or sys.argv[1] not in valid_conf:
        print(
            "Falling back on yolov8l model. Configuration was not passed or was not valid."
        )
        print(
            "If you want to change configuration, give as argument one valid option between {n, s, m, l, x}."
        )
        conf = "l"
    else:
        conf = str(sys.argv[1])

    download_variant(conf)
    download_filename = f"yolov8{conf}.pt"

    model = YOLOv8(*get_variant_multiples(conf))
    model_dict_keys = list(model.state_dict().keys())
    model_dict = model.state_dict()
    downloaded_dict = torch.load(download_filename)["model"].state_dict()
    downloaded_dict_keys = downloaded_dict.keys()

    os.remove(download_filename)

    for i, key in enumerate(downloaded_dict_keys):
        assert model_dict[model_dict_keys[i]].shape == downloaded_dict[key].shape
        if downloaded_dict[key].size != 1:
            tmp = torch.tensor(downloaded_dict[key], dtype=downloaded_dict[key].dtype)
        else:
            tmp = torch.tensor([downloaded_dict[key]], dtype=downloaded_dict[key].dtype)
        model_dict[model_dict_keys[i]] = tmp
        assert (
            model_dict[model_dict_keys[i]] == downloaded_dict[key]
        ).all(), f"Failed at: {model_dict_keys[i]}."

    torch.save(model_dict, download_filename)
