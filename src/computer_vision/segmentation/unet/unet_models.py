import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import math
import time
import gc
from typing import Union
from abc import ABC, abstractmethod



class ModelBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

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
        images, mask = batch
        mask_pred = self(images)

        train_loss = self.loss(mask_pred, mask)


        return train_loss

    def validation_step(self, batch):
        self.eval()
        with torch.no_grad():
            images, mask = batch
            mask_pred = self(images)
            val_loss = self.loss(mask_pred, mask)

            return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        Returns the epoch losses after computing the mean loss and accuracy of the test batches

        :param outputs: List of test step outputs
        :type outputs: list
        :return: epoch loss and epoch accuracy
        :rtype: dict
        """
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean().item()

        return {"val_loss": epoch_loss}

    def evaluate(self, dl):
        outputs = [self.validation_step(batch) for batch in dl]
        return self.validation_epoch_end(outputs)

    def epoch_end_val(self, epoch, results):
        """
        Prints validation epoch summary after every epoch

        :param epoch: epoch number
        :type epoch: int
        :param results: results from the evaluate method
        :type results: dictionary
        :return: None
        """

        print(
            f"Epoch:[{epoch}]: |validation loss: {results['val_loss']}|"
        )

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2), (diffY // 2, diffY - diffY // 2))

        else:
            x = x1

        x = self.conv(x)
        return(x)


class HeatMapSkipConnections512(nn.Module, ModelBase):
    def __init__(self, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone: bool = True):
        """
        HeatMapCenterNet class.

        :param final_resolution: The final resolution of the heatmap
        :param num_classes: The number of output classes. Defaults to 1.
        :param model_name: The name of the backbone model. Defaults to "resnet18".
        """
        nn.Module.__init__(self)

        basemodel = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.basemodel = nn.Sequential(*list(basemodel.children())[:-2])

        for param in self.basemodel.parameters():
            param.requires_grad = train_backbone

        self._NUM_CLASSES = num_classes
        self.alpha = alpha
        self.beta = beta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if class_weights is not None:
            self.class_weights = torch.tensor(list(class_weights.values()))

        # Instantiate hooks
        # Keep the outputs of the intermediate layers
        self.outputs = {}

        for i, layer in enumerate(list(self.basemodel.children())):
            layer.register_forward_hook(self.save_output)

        self.maxpool = nn.MaxPool2d(2)

        # Create decoding part
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up6 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.up_conv4 = DoubleConv(256 + 256, 256)
        self.up_conv3 = DoubleConv(2 + 128, 128)
        self.up_conv3 = DoubleConv(128 + 128, 128)
        self.up_conv2 = DoubleConv(64 + 64, 64)
        self.up_conv1 = DoubleConv(32 + 64, 64)
        self.up_conv0 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, self._NUM_CLASSES, kernel_size=1)

    def save_output(self, module, input, output):
        self.outputs[module] = output

    def forward(self, x):
        self.basemodel(x)

        x = self.up2(self.outputs[list(self.basemodel.children())[-1]])
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-2]]), dim=1)
        x = self.up_conv4(x)

        x = self.up3(x)
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-3]]), dim=1)
        x = self.up_conv3(x)

        x = self.up4(x)
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-4]]), dim=1)
        x = self.up_conv2(x)

        x = self.up5(x)
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-6]]), dim=1)
        x = self.up_conv1(x)

        x = self.up6(x)
        x = self.up_conv0(x)

        out = self.outc(x)

        return out


class Unet32(nn.Module, ModelBase):
    def __init__(self, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone: bool = True):
        """
        HeatMapCenterNet class.

        :param final_resolution: The final resolution of the heatmap
        :param num_classes: The number of output classes. Defaults to 1.
        :param model_name: The name of the backbone model. Defaults to "resnet18".
        """
        nn.Module.__init__(self)

        basemodel = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.basemodel = nn.Sequential(*list(basemodel.children())[:-2])

        for param in self.basemodel.parameters():
            param.requires_grad = train_backbone

        self._NUM_CLASSES = num_classes
        self.alpha = alpha
        self.beta = beta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if class_weights is not None:
            self.class_weights = torch.tensor(list(class_weights.values()))

        # Instantiate hooks
        # Keep the outputs of the intermediate layers
        self.outputs = {}

        for i, layer in enumerate(list(self.basemodel.children())):
            layer.register_forward_hook(self.save_output)

        self.maxpool = nn.MaxPool2d(2)

        # Create decoding part
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.up_conv4 = DoubleConv(256 + 256, 256)
        self.up_conv3 = DoubleConv(2 + 128, 128)
        self.up_conv3 = DoubleConv(128 + 128, 128)
        self.up_conv2 = DoubleConv(64 + 64, 64)
        self.up_conv1 = DoubleConv(32 + 64, 64)

        self.outc3 = nn.Conv2d(128, self._NUM_CLASSES, kernel_size=1)
        self.outc2 = nn.Conv2d(64, self._NUM_CLASSES, kernel_size=1)
        self.outc1 = nn.Conv2d(64, self._NUM_CLASSES, kernel_size=1)


    def save_output(self, module, input, output):
        self.outputs[module] = output

    def forward(self, x):
        self.basemodel(x)

        x = self.up2(self.outputs[list(self.basemodel.children())[-1]])
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-2]]), dim=1)
        x = self.up_conv4(x)

        x = self.up3(x)
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-3]]), dim=1)
        x = self.up_conv3(x)
        output_first_map = nn.Upsample(scale_factor=8, mode="bilinear")(self.outc3(x))

        x = self.up4(x)
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-4]]), dim=1)
        x = self.up_conv2(x)
        intermediate_output_second = nn.Upsample(scale_factor=4, mode="bilinear")(self.outc2(x))
        output_second_map = torch.add(output_first_map, intermediate_output_second)

        x = self.up5(x)
        x = torch.cat((x, self.outputs[list(self.basemodel.children())[-6]]), dim=1)
        x = self.up_conv1(x)
        intermediate_output_third = nn.Upsample(scale_factor=2, mode="bilinear")(self.outc1(x))
        final_output = torch.add(output_second_map, intermediate_output_third)

        return final_output


def focal_loss_segmentation(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 3, beta: float = 4,
                       weights: Union[torch.Tensor, None] = None, verbose: bool = False) -> torch.Tensor:
    """
    Computes focal loss for heatmap prediction. Based on https://arxiv.org/pdf/1808.01244.pdf (see page 5)

    :param pred: prediction tensor
    :param gt: ground thruth tensor
    :param alpha: alpha hyperparameter of loss
    :param beta: beta hyperparameter of loss for the negative weight (the higher the less impact have negative samples)
    :param weights: class weights tensor
    :param verbose:
    :return:
    """

    if weights is not None:
        weights = weights.to(pred.device)
        try:
            weights = weights.view(1, pred.shape[1], 1, 1)
        except RuntimeError:
            raise ValueError("Bitte sicherstelle, dass in der Config nur die class weights für Klassen angegeben sind, "
                             "auf die auch trainiert wird!")



    # convert to probabilities
    pred_sigmoid = torch.sigmoid(pred)  # enable during training

    # Finding where the ground truth is positive (i.e., = 1)
    pos_inds = gt.eq(1).float()

    # Finding where the ground truth is negative (i.e., < 1)
    neg_inds = gt.lt(1).float()

    # TODO: compare num of pos indices to num of neg indices and tackle the class imbalance!!
    # Initialize loss to 0
    loss = 0

    # Calculate the negative weights
    neg_weights = torch.pow(1 - gt, beta)

    # Compute positive loss
    # WICHTIG: Ich habe 1e-7 verwendet, weil ab 1e-8 und der Verwendung von Mixed Precision für floating point numbers
    # die 1nur 16bit verwenden, dies nicht mehr berechnet werden kann und einfach als 0 ausgegeben wird
    # Trade off zwischen genaugigkeit und effizienz sehe ich effizienz vor, weil wir mit 1e-7 trotzdem noch echt gut
    # approximieren können und den loss nicht wirklich verzerren

    try:
    #TODO: aktuell sind predictions für center points oft so bei 0.1 - 0.2 --> hier sollte das modell noch sicherer werden
        pos_loss = torch.pow(1 - pred_sigmoid, alpha) * torch.log(pred_sigmoid + 1e-12) * pos_inds
    except RuntimeError:
        raise ValueError("Bitte prüfen ob Model Scale passt in der config file!")

    # Compute negative loss
    neg_loss = neg_weights * torch.pow(pred_sigmoid, alpha) * torch.log(1 - pred_sigmoid + 1e-7) * neg_inds

    # Count the number of positive and negative samples
    num_pos = pos_inds.float().sum()

    if weights is not None:
        pos_loss = pos_loss * weights
        neg_loss = neg_loss * weights

    if verbose:
        for i in range(pos_loss.shape[1]):
            try:
                print(f"Positive loss label {i}: {-pos_loss[i].sum()}")
                print(f"Negative loss label {i}: {-neg_loss[i].sum()}")
            except IndexError:
                print(f"Negative loss label {i}: {-neg_loss[i].sum()}")

    if num_pos == 0:
        loss = -neg_loss.sum()
    else:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    if torch.isnan(loss):
        print("is nan")

    return loss