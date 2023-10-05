from micromind import MicroMind, MicroTrainer
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)


class ImageClassification(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules.append(
            PhiNet(
                (3, 32, 32),
                include_top=True,
                num_classes=10
            )
        )

        self.modules.to(self.device)

    def forward(self, batch):
        x, _ = batch
        x = x.to(self.device)
        return self.modules[0](x)

    def compute_loss(self, pred, batch):
        pred = pred.to(self.device)
        labels = batch[1].to(self.device)
        return nn.CrossEntropyLoss()(pred, labels)


if __name__ == "__main__":
    m = ImageClassification()
    optimizer = m.configure_optimizers()

    trainer = MicroTrainer(
        m,
        epochs=10,
        datasets={"train": trainloader, "val": testloader},
        device="mps",
        debug=True
    )

    trainer.train()


