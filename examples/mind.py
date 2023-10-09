from micromind import MicroMind
from micromind import PhiNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
        [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='/mnt/data/cifar-10', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='/mnt/data/cifar-10', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)


class ImageClassification(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["classifier"] = PhiNet(
                (3, 32, 32),
                include_top=True,
                num_classes=10
            )

    def forward(self, batch):
        images = batch[0].to(self.device)
        return self.modules["classifier"](images)

    def compute_loss(self, pred, batch):
        labels = batch[1].to(self.device)
        return nn.CrossEntropyLoss()(pred, labels)


if __name__ == "__main__":
    m = ImageClassification()

    m.train(
        epochs=10,
        datasets={"train": trainloader, "val": testloader, "test": testloader},
        debug=False
    )

    m.test(
        datasets={"test": testloader},
    )

