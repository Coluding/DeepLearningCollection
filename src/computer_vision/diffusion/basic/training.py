import torch
from torch.utils.data import Dataset, DataLoader
from backward_process import *
from sampling import *
from noise_scheduler import *
from typing import Callable
from tqdm import tqdm


class DiffusionPipeline:
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 T: int = 300,
                 batch_size: int = 32,
                 loss: Callable = None,
                 device: str = "cpu"):

        self.betas = linear_beta_scheduler(timesteps=T)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.T = T
        self.device = device
        self.img_size = None
        self.train_tracker = {}
        self.train_tracker["loss"] = []

    def train(self, epochs: int):
        epoch_iterator = tqdm(range(epochs), desc="Training")
        best_loss = float("inf")
        for epoch in epoch_iterator:
            batch_iterator = tqdm(enumerate(self.loader), desc=f"Epoch {epoch}")

            epoch_loss = 0
            for step, batch in batch_iterator:
                batch = batch.to(self.device)
                if self.img_size is None:
                    self.img_size = batch.shape[-2]
                t = torch.randint(0, self.T, (batch.shape[0],)).long().to(self.device)
                self.optimizer.zero_grad()
                loss = self._loss(batch, t)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()


            epoch_loss /= len(self.loader)
            self.train_tracker["loss"].append(epoch_loss)
            if epoch % 2 == 0:
                print(f"Epoch {epoch} completed with loss {epoch_loss}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), "best_model.pth")


        print("Training completed.")
        self.visualize_training()
        self.visual_validation()
        return self.train_tracker

    def visual_validation(self, num_images: int = 10, img_size: int = None, num_processes=1):
        if self.img_size is None and img_size is None:
            raise ValueError("Please train the model first or insert image size as argument.")

        sample_plot_image(
            self.img_size if img_size is None else img_size,
            self.T,
            self.device,
            self.model,
            self.betas,
            self.sqrt_one_minus_alphas_cumprod,
            self.sqrt_recip_alphas,
            self.posterior_variance,
            num_images=num_images,
            num_processes=num_processes
        )

    def visualize_training(self):
        plt.plot(self.train_tracker["loss"])
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


    def _loss(self, x_0, t):
        if self.loss is not None:
            return self.loss(x_0, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.device)

        x_noisy, noise = forward_diffusion_sample(x_0, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.device)
        noise_pred = self.model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def load_state(self, path: str):
        self.model.load_state_dict(torch.load(path))