import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from typing import Union

from src.computer_vision.model_utils import ModelBase


class AttentionGate(nn.Module):
    def __init__(self, in_channels_g, in_channels_x=None):
        super().__init__()

        if in_channels_x is None:
            in_channels_x = in_channels_g // 2

        self.x_tr = nn.Conv2d(in_channels_x, in_channels_g, kernel_size=1, stride=2)
        self.g_tr = nn.Conv2d(in_channels_g, in_channels_g, kernel_size=1, stride=1)
        self.weights = nn.Conv2d(in_channels_g, 1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, g, x):
        x_tr = self.x_tr(x)
        g_tr = self.g_tr(g)
        aligned_weights = torch.add(x_tr, g_tr)
        aligned_weights = torch.relu(aligned_weights)

        weights = self.weights(aligned_weights)
        weights = torch.sigmoid(weights)
        weights = self.upsample(weights)

        return weights


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


class UNet512(nn.Module, ModelBase):
    def __init__(self, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone: bool = True):
        """
        UNet withou attention gate.

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
        Unet32 class without attention gates.

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



class AttentionUNetSegmentation(nn.Module, ModelBase):
    def __init__(self, loss: nn.Module, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone: bool = True, **kwargs):
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
        self.up_conv3 = DoubleConv(128 + 128, 128)
        self.up_conv2 = DoubleConv(64 + 64, 64)
        self.up_conv1 = DoubleConv(32 + 64, 64)

        self.outc3 = nn.Conv2d(128, self._NUM_CLASSES, kernel_size=1)
        self.outc2 = nn.Conv2d(64, self._NUM_CLASSES, kernel_size=1)
        self.outc1 = nn.Conv2d(64, self._NUM_CLASSES, kernel_size=1)

        self.attention1 = AttentionGate(512)
        self.attention2 = AttentionGate(256)
        self.attention3 = AttentionGate(128)
        self.attention4 = AttentionGate(64, 64)

        self.loss = loss


    def save_output(self, module, input, output):
        self.outputs[module] = output

    def forward(self, x):
        self.basemodel(x)

        x = self.up2(self.outputs[list(self.basemodel.children())[-1]])
        skip_conn = self.outputs[list(self.basemodel.children())[-2]]
        attention_skip = self.attention1(self.outputs[list(self.basemodel.children())[-1]],
                                         skip_conn)

        weighted_skip_conn = skip_conn * attention_skip
        x = torch.cat((x, weighted_skip_conn), dim=1)
        x_ = self.up_conv4(x)

        x = self.up3(x_)
        skip_conn = self.outputs[list(self.basemodel.children())[-3]]
        attention_skip = self.attention2(x_, skip_conn)
        weighted_skip_conn = torch.matmul(skip_conn, attention_skip)
        x = torch.cat((x, weighted_skip_conn), dim=1)
        x_ = self.up_conv3(x)
        output_first_map = nn.Upsample(scale_factor=8, mode="bilinear")(self.outc3(x_))

        x = self.up4(x_)
        skip_conn = self.outputs[list(self.basemodel.children())[-4]]
        attention_skip = self.attention3(x_, skip_conn)
        weighted_skip_conn = skip_conn * attention_skip
        x = torch.cat((x, weighted_skip_conn), dim=1)
        x_ = self.up_conv2(x)
        intermediate_output_second = nn.Upsample(scale_factor=4, mode="bilinear")(self.outc2(x_))
        output_second_map = torch.add(output_first_map, intermediate_output_second)

        x = self.up5(x_)
        skip_conn = self.outputs[list(self.basemodel.children())[-6]]
        attention_skip = self.attention4(x_, skip_conn)
        weighted_skip_conn = skip_conn * attention_skip
        x = torch.cat((x, weighted_skip_conn), dim=1)
        x_ = self.up_conv1(x)
        intermediate_output_third = nn.Upsample(scale_factor=2, mode="bilinear")(self.outc1(x_))
        final_output = torch.add(output_second_map, intermediate_output_third)

        return final_output

