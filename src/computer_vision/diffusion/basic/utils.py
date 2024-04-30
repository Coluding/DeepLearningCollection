import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from noise_scheduler import *
import numpy as np
from typing import List

from data import StanfordCars



def load_transformed_dataset(train_path="data/cars_train/cars_train",
                             test_path="data/cars_test/cars_test",
                             img_size=64):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = StanfordCars(train_path, transform=data_transform)
    test = StanfordCars(test_path, transform=data_transform)

    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
    """
    Applies reverse transformations to a tensor image and displays it
    :param image:
    :return:
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    plt.axis('off')
    plt.show()



def show_noising_process(x: List[List[torch.tensor]]):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if isinstance(x[0], torch.Tensor):
        x = [x]

    transformed_x = [[reverse_transforms(img) for img in sublist] for sublist in x]

    # Calculate the aspect ratio of a single image
    aspect_ratio = transformed_x[0][0].size[1] / transformed_x[0][0].size[0]

    # Set up the figure size based on the number and aspect ratio of images
    fig_width = len(transformed_x) * 2  # 2 inches per image
    fig_height = fig_width * aspect_ratio  # Maintain the aspect ratio for the height

    # Create the figure with the calculated dimensions
    fig, axes = plt.subplots(len(transformed_x), len(transformed_x[0]), figsize=(fig_width, fig_height))

    for i, row in enumerate(transformed_x):
        for j, img in enumerate(row):
            if len(transformed_x) == 1 or len(transformed_x[0]) == 1:  # If there's only one row or column
                ax = axes[i] if len(transformed_x) == 1 else axes[j]
            else:
                ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between images
    plt.tight_layout(pad=2)
    plt.show()





