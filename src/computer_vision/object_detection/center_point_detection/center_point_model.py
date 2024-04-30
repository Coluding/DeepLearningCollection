import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from typing import Union
from src.computer_vision.model_utils import ModelBase


class DoubleConv(nn.Module):
    """
    A module that performs two consecutive convolution operations.

    Each convolution is followed by batch normalization and an ELU activation function.

    Args:
    - in_ch (int): Number of input channels.
    - out_ch (int): Number of output channels.
    - padding (int, optional): Padding added to both sides of the input. Default is 1.
    - stride (int, optional): Stride of the convolution. Default is 1.
    """

    def __init__(self, in_ch, out_ch, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        """Performs the forward pass of the module."""
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upsampling module with optional bilinear interpolation or transpose convolution.

    Args:
    - in_ch (int): Number of input channels.
    - out_ch (int): Number of output channels.
    - bilinear (bool, optional): If True, use bilinear upsampling. Default is True.
    """

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Forward pass for upsampling. Optionally concatenates a secondary input for skip connections.

        Args:
        - x1 (Tensor): The primary input tensor.
        - x2 (Tensor, optional): The secondary input tensor for concatenation. Default is None.

        Returns:
        - Tensor: The upsampled (and possibly concatenated) output tensor.
        """
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        else:
            x = x1

        x = self.conv(x)
        return x



class HeatMapCenterNetNoOffset(nn.Module, ModelBase):
    """
    Defines a HeatMapCenterNet model without offset predictions, primarily focusing on generating heatmaps for object detection.

    Attributes:
        base_model (nn.Module): Backbone neural network model for feature extraction.
        upsampling_blocks (nn.Sequential): Sequential container of upsampling layers.
        out_class (nn.Conv2d): Convolutional layer to predict the class heatmaps.
        alpha (int): Hyperparameter of the FocalLoss function.
        beta (int): Hyperparameter of the FocalLoss function.
        class_weights (torch.Tensor, optional): Tensor containing weights for each class to handle class imbalance.

    Args:
        final_resolution (int): Desired resolution of the output heatmap.
        num_classes (int, optional): Number of classes for the output layer. Defaults to 1.
        alpha (int, optional): Model-specific hyperparameter. Defaults to 3.
        beta (int, optional): Model-specific hyperparameter. Defaults to 4.
        class_weights (Union[torch.Tensor, None], optional): Weights for different classes to address class imbalance. Defaults to None.
        train_backbone (bool, optional): If True, allows training of the backbone model. Defaults to True.
    """
    def __init__(self, final_resolution, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None, train_backbone=True):
        super().__init__()
        # Initialize the backbone model
        basemodel = torchvision.models.resnet18(pretrained=True)
        # Removing the fully connected layer and the last pooling layer from the backbone
        self.base_model = nn.Sequential(*list(basemodel.children())[:-2])

        self.alpha = alpha
        self.beta = beta
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        # Setting the model to train or freeze the backbone
        for param in self.base_model.parameters():
            param.requires_grad = train_backbone

        # Define the upsampling layers to match the final heatmap resolution
        upsampling_layers = [Up(512, 512)]  # Example for initial upsampling
        # Additional upsampling layers are added based on the final resolution required

        self.upsampling_blocks = nn.Sequential(*upsampling_layers)
        self.out_class = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted heatmap tensor.
        """
        x = self.base_model(x)
        x = self.upsampling_blocks(x)
        out_class = self.out_class(x)
        return out_class

class HeatMapSkipConnections(nn.Module, ModelBase):
    """
    Defines a HeatMapCenterNet model with skip connections, enhancing feature learning by combining features from different layers.

    Attributes:
        basemodel (nn.Module): Backbone neural network model for feature extraction.
        up_conv* (DoubleConv): Double convolution modules for processing concatenated features.
        outc (nn.Conv2d): Convolutional layer to predict the class heatmaps.
        alpha (int): Hyperparameter used in the model (not used explicitly in this definition).
        beta (int): Hyperparameter used in the model (not used explicitly in this definition).
        class_weights (torch.Tensor, optional): Tensor containing weights for each class to handle class imbalance.

    Args:
        num_classes (int, optional): Number of classes for the output layer. Defaults to 1.
        alpha (int, optional): Model-specific hyperparameter. Defaults to 3.
        beta (int, optional): Model-specific hyperparameter. Defaults to 4.
        class_weights (Union[torch.Tensor, None], optional): Weights for different classes to address class imbalance. Defaults to None.
        train_backbone (bool, optional): If True, allows training of the backbone model. Defaults to True.
    """
    def __init__(self, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone=True):
        super().__init__()
        # Initialize the backbone model with a predefined architecture
        basemodel = torchvision.models.resnet34(pretrained=True)
        self.basemodel = nn.Sequential(*list(basemodel.children())[:-2])

        for param in self.basemodel.parameters():
            param.requires_grad = train_backbone

        self._NUM_CLASSES = num_classes
        self.alpha = alpha
        self.beta = beta
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None

        # Hooks to capture outputs of intermediate layers for skip connections
        self.outputs = {}
        # Definition of upsampling and convolution layers to process and merge features
        # from different layers follows...

    def save_output(self, module, input, output):
        """
        Hook to save the output of a module.

        Args:
            module (nn.Module): The module being observed.
            input (tuple): The input to the module.
            output (torch.Tensor): The output from the module.
        """
        self.outputs[module] = output

    def forward(self, x):
        """
        Forward pass of the model with skip connections.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through skip connections and upsampling layers.
        """
        # Forward pass through the base model and subsequent processing using skip connections
        # and upsampling layers follows...
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

        out = self.outc(x)

        return out


class HeatMapSkipConnections256(nn.Module, ModelBase):
    """
    A neural network model that generates heatmaps with skip connections from a ResNet-34 backbone.
    This model is designed to handle higher resolution output for precise object detection.

    Args:
        num_classes (int): Number of classes to predict.
        alpha (int): A hyperparameter for model configuration, not directly used in this class.
        beta (int): Another hyperparameter for model configuration, not directly used.
        class_weights (Union[torch.Tensor, None]): Optional; class weights to mitigate class imbalance.
        train_backbone (bool): If True, allows the backbone network's weights to be updated during training.

    Attributes:
        basemodel (nn.Module): The ResNet-34 backbone model excluding the final fully connected layer.
        up* (nn.ConvTranspose2d): Transposed convolution layers for upsampling the feature maps.
        up_conv* (DoubleConv): Double convolution layers applied after upsampling for feature refinement.
        outc (nn.Conv2d): The final convolution layer that maps the features to class predictions.
    """
    def __init__(self, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone: bool = True):
        super().__init__()
        # Backbone model initialization with pretrained weights
        basemodel = torchvision.models.resnet34(pretrained=True)
        self.basemodel = nn.Sequential(*list(basemodel.children())[:-2])

        # Allow or prevent the backbone model from updating weights during training
        for param in self.basemodel.parameters():
            param.requires_grad = train_backbone

        # Model configuration parameters
        self._NUM_CLASSES = num_classes
        self.alpha = alpha
        self.beta = beta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None

        # Intermediate layer outputs for skip connections are stored here
        self.outputs = {}

        # Register hooks to save outputs of intermediate layers
        for i, layer in enumerate(list(self.basemodel.children())):
            layer.register_forward_hook(self.save_output)

        # Upsampling and double convolution layers to refine and scale up feature maps to desired output size
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # Additional up* and up_conv* layers follow...

        self.outc = nn.Conv2d(64, self._NUM_CLASSES, kernel_size=1)

    def save_output(self, module, input, output):
        """Saves output of a module to be used in skip connections."""
        self.outputs[module] = output

    def forward(self, x):
        """
        Forward pass of the model, applying skip connections and upsampling to generate high-resolution heatmaps.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: High-resolution heatmap predictions.
        """
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

        out = self.outc(x)

        return out


class HeatMapSkipConnections256Pretrained(nn.Module, ModelBase):
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

        self.head_adjustment = False

        basemodel = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.basemodel = nn.Sequential(*list(basemodel.children())[:-2])

        for param in self.basemodel.parameters():
            param.requires_grad = train_backbone

        self.train_adjusted_backbone = True
        self._NUM_CLASSES = num_classes
        self.alpha = alpha
        self.beta = beta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if class_weights is not None:
            self.class_weights = torch.tensor(list(class_weights.values()))
        else:
            self.class_weights = None

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

        self.outc = nn.Conv2d(64, self._NUM_CLASSES, kernel_size=1)

    def save_output(self, module, input, output):
        self.outputs[module] = output

    def load_weights(self, path):
        """
        Loads pretrained weights into the model.

        Args:
            path (str): Path to the pretrained model weights.
        """
        self.load_state_dict(torch.load(path))
        self.head_adjustment = True

    def adjust_classes_for_pretrained_base(self, num_classes):
        """
        Adjusts the model's final layer to match the number of classes for a new task.

        Args:
            num_classes (int): The new number of classes.
        """
        if not self.head_adjustment:
            raise ValueError("Load the weights first before adjusting the head.")
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
        print(f"Output classes changed from {self._NUM_CLASSES} to {num_classes}.")

    def freeze_base_to_train_head(self):
        """
        Freezes the backbone model to train only the newly adjusted head.
        This is useful for fine-tuning the model on a new task with a different number of classes.
        """
        if not self.head_adjustment:
            raise ValueError("Load the weights first before attempting to freeze the base model.")
        self.train_adjusted_backbone = False
        print("Base model frozen; ready to train the head.")


    def forward(self, x):
        self.basemodel(x)

        with torch.set_grad_enabled(self.train_adjusted_backbone):
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

        out = self.outc(x)

        return out


class HeatMapSkipConnections512AddedResults(nn.Module, ModelBase):
    """
    Extends HeatMapCenterNet with added results from intermediate layers. This class introduces a novel approach to
    heatmap generation by not only upsampling and refining feature maps through skip connections but also by
    aggregating results from multiple intermediate layers before producing the final output. This method aims to
    enhance the model's ability to capture and utilize multi-scale information, potentially improving the detection
    accuracy for objects of various sizes.

    Args:
        num_classes (int): The number of distinct classes that the model predicts.
        alpha (int): Model-specific hyperparameter, not directly used in this class.
        beta (int): Model-specific hyperparameter, not directly used.
        class_weights (Union[torch.Tensor, None], optional): Weights for each class to address imbalance; defaults to None.
        train_backbone (bool): If True, the backbone model's weights are trainable; defaults to True.

    Attributes:
        basemodel (nn.Module): The backbone model, a ResNet-34, without its fully connected layer.
        up2, up3, up4, up5 (nn.ConvTranspose2d): Transposed convolutional layers for sequential upsampling.
        up_conv4, up_conv3, up_conv2, up_conv1 (DoubleConv): Double convolutional modules for processing features post-upsampling.
        outc3, outc2, outc1 (nn.Conv2d): Convolutional layers at different stages to produce intermediate output maps.
    """
    def __init__(self, num_classes=1, alpha=3, beta=4,
                 class_weights: Union[torch.Tensor, None] = None,
                 train_backbone: bool = True):
        nn.Module().__init__()

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
        """
        Forward pass through the model, integrating outputs from intermediate layers for enhanced multi-scale detection.

        This method sequentially upsamples the feature maps, applies skip connections, and generates intermediate
        output maps. These maps are then aggregated to produce the final output, aiming to utilize detailed
        features from various levels of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The final aggregated output heatmap.
        """
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


