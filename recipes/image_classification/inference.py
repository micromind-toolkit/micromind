"""
This code runs the image classification training loop. It tries to support as much
as timm's functionalities as possible.

For compatibility the prefetcher, re_split and JSDLoss are disabled.

To run the training script, use this command:
    python train.py cfg/phinet.py

You can change the configuration or override the parameters as you see fit.

Authors:
    - Francesco Paissan, 2023
"""

import sys
import torch
from train import ImageClassification
from micromind.utils import parse_configuration
import torchvision


class ImageClassification(ImageClassification):
    """Implements an image classification class for inference."""

    def forward(self, batch):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted logits.
        """
        return self.modules["classifier"](batch[0])

    def compute_loss(self, pred, batch):
        """Ignoring because it's inference."""
        pass

    def configure_optimizers(self):
        """Ignoring because it's inference."""
        pass


def top_k_accuracy(k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """

    def acc(pred, batch):
        if pred[1].ndim == 2:
            target = pred[1].argmax(1)
        else:
            target = pred[1]
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        accuracy = correct.item() / target.size(0)

        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    mind = ImageClassification(hparams=hparams)
    if hparams.ckpt_pretrained != "":
        mind.load_modules(hparams.ckpt_pretrained)
    mind.eval()

    # read, resize, and normalize image
    img = torchvision.io.read_image(sys.argv[2])
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=hparams.input_shape[1:]),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    img = preprocess(img.float() / 255)
    logits = mind((img[None],))

    print(
        "Model prediction: %d with probability: %.2f."
        % (logits.argmax(1).item(), logits.softmax(1)[0, logits.argmax(1)].item())
    )

    print("Saving exported model to model.onnx...")
    mind.export("model.onnx", "onnx", (3, 32, 32))
