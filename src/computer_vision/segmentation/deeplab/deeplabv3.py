import torch.nn as nn
import torch
import torchvision
import time
import gc

from src.computer_vision.segmentation.deeplab.aspp import ASPP
from src.computer_vision.model_utils import ModelBase
from src.computer_vision.segmentation.segmentation_utils import SegmentationModelUtils as ModelUtils


class DeepLabv3Plus(nn.Module, ModelBase):
    def __init__(self, loss, num_classes: int = 4,  final_softmax=True):
        super().__init__()
        backbone = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.DEFAULT
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-3])
        self.aspp = ASPP(1024, 256)
        self.onebyone_conv = nn.Conv2d(1024, 256, 1, padding="same")
        self.low_level_upsampler = nn.Upsample(scale_factor=4, mode="bilinear")
        self.up_encoder = nn.Upsample(scale_factor=4, mode="bilinear")
        self.up_backbone = nn.Upsample(scale_factor=4, mode="bilinear")
        self.decoder_conv = nn.Conv2d(512, num_classes, 3, padding="same")
        self.up_final = nn.Upsample(scale_factor=4, mode="bilinear")

        if final_softmax:
            self.sm = nn.Softmax2d()
        else:
            self.sm = nn.Identity()

        self.loss = loss

    def forward(self, x):
        x = self.backbone(x)
        low_level_features = self.onebyone_conv(x)
        low_level_features_up = self.low_level_upsampler(low_level_features)
        encoder_out = self.aspp(x)
        encoder_out_up = self.up_encoder(encoder_out)

        combined = torch.cat((low_level_features_up, encoder_out_up), dim=1)
        out = self.decoder_conv(combined)
        out = self.sm(out)
        up = self.up_final(out)

        return up

    def training_step(self, batch):
        """
        Training step which computes the loss and accuracy of a train batch
        :param batch: batch of pytorch dataloader
        :type batch: torch.utils.data.DataLoader
        :return: loss, accuracy and f1_score of batch
        :rtype: tuple[torch.tensor,...]
        """
        # Runs the forward pass with autocasting.
        #with torch.cuda.amp.autocast():
        images, heatmap_target = batch
        heatmap_pred = self(images)

        train_loss = self.loss(heatmap_pred, heatmap_target)
        dices = {i: ModelUtils.dice_score(heatmap_pred[i], heatmap_target[i]) for i in range(len(heatmap_target))}
        del images
        del heatmap_target
        gc.collect()
        print("train_dice:")
        print(dices)
        return train_loss

    def validation_step(self, batch):
        self.eval()
        with torch.no_grad():
            images, heatmap_target = batch
            heatmap_pred = self(images)
            val_loss = self.loss(heatmap_pred, heatmap_target)
            dices = {i: ModelUtils.dice_score(heatmap_pred[i], heatmap_target[i]) for i in range(len(heatmap_target))}

            del images
            del heatmap_target
            gc.collect()
            print("val dice:")
            print(dices)
            return {"val_loss": val_loss}

