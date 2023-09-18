import torch
import torch.nn as nn

from .yolo_losses import Yolov1Loss


class ReusableConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(mid_ch, out_ch, 3, padding="same"),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Yolov1(nn.Module):
    def __init__(self):
        super().__init__()
        reusable1 = ReusableConvBlock(512, 256, 512)
        reusable2 = ReusableConvBlock(512, 256, 512)
        reusable3 = ReusableConvBlock(512, 256, 512)
        reusable4 = ReusableConvBlock(512, 256, 512)

        reusable5 = ReusableConvBlock(1024, 512, 1024)
        reusable6 = ReusableConvBlock(1024, 512, 1024)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 192, 7, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(192))

        self.block2 = nn.Sequential(
            nn.Conv2d(192, 256, 3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256))

        self.block3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
        )
        self.block4 = nn.Sequential(
            reusable1,
            nn.BatchNorm2d(512),
            reusable2,
            nn.BatchNorm2d(512),
            reusable3,
            nn.BatchNorm2d(512),
            reusable4,
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, 3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(1024),
        )

        self.block5 = nn.Sequential(
            reusable5,
            nn.BatchNorm2d(1024),
            reusable6,
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 4096),
            nn.ReLU(),
            nn.Linear(4096, 441),
            nn.Sigmoid()
        )

        self.loss = Yolov1Loss(4, 2)

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return self.fc(x)

    def training_step(self, batch):
        self.train()
        img, label = batch
        out = self(img)
        out = out.reshape(-1, 7, 7, 9)
        loss = self.loss(out, label)
        return loss

    def validation_step(self, batch):
        self.eval()
        with torch.no_grad():
            img, label = batch
            out = self(img)
            out = out.reshape(-1, 7, 7, 9)
            loss = self.loss(out, label)

            return loss

    def validation_epoch_end(self, outputs: list):
        batch_losses = torch.stack(outputs)
        epoch_loss = torch.mean(batch_losses)

        return {"val_loss": epoch_loss}

    def evaluate(self, dl):
        outputs = [self.validation_step(batch) for batch in dl]
        return self.validation_epoch_end(outputs)

    def epoch_end_val(self, epoch: int, results: dict):
        print(f"Epoch [{epoch}]: Loss {results['val_loss']}")
