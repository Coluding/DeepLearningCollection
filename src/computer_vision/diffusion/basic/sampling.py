import matplotlib.pyplot as plt
from noise_scheduler import *
from utils import *
from tqdm import tqdm

@torch.no_grad()
def sample_timestep(x: torch.Tensor,
                    t: torch.Tensor,
                    model: torch.nn.Module,
                    betas: torch.Tensor,
                    sqrt_one_minus_alphas_cumprod: torch.Tensor,
                    sqrt_recip_alphas: torch.Tensor,
                    posterior_variance: torch.Tensor):
    """
    Perform a single timestep of the reverse diffusion process on input data.

    This function takes a noisy input tensor and a specific timestep, then applies
    the model to estimate and subtract the noise, effectively denoising the input
    data as part of the reverse diffusion process. For each timestep, it computes
    the necessary parameters to adjust the data, optionally adding a scaled noise
    based on the posterior variance to simulate the reverse diffusion.

    Parameters:
    - x (torch.Tensor): The current noisy input data at timestep t. This is the tensor
        being progressively denoised over the diffusion steps.
    - t (torch.Tensor): The current timestep in the reverse diffusion process, indicating
        the step at which the denoising is to be applied.
    - model (torch.nn.Module): The neural network model trained to predict the noise
        added to the original data, used for reversing the diffusion process. The output shape of the model should be
        HxWX3 for images.
    - betas (torch.Tensor): A tensor containing the β (beta) parameters for each timestep
        in the diffusion process, controlling the amount of noise added during the forward
        process.
    - sqrt_one_minus_alphas_cumprod (torch.Tensor): A tensor with the square root of
        one minus the cumulative product of α (alpha) values, derived from betas and used
        in denoising calculations.
    - sqrt_recip_alphas (torch.Tensor): A tensor containing the square root of the reciprocal
        of alphas, involved in adjusting the data during the denoising process.
    - posterior_variance (torch.Tensor): A tensor representing the variance of the posterior
        distribution at each timestep, used to scale the noise added during sampling.

    Returns:
    - torch.Tensor: The denoised data at the given timestep. For t=0, it returns the model
        mean directly without adding noise. For timesteps > 0, it adds scaled Gaussian noise
        to the model mean, simulating the reverse diffusion process and returns the less noisy image.

    Note:
    This function should be called with `torch.no_grad()` context to prevent tracking of
    gradients, as it is used during the inference phase of the diffusion model.
    """

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # image - noise_prediction
    # sampling algorithm from the paper
    # it subtracts the predicted noise from the image
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x,t) / sqrt_one_minus_alphas_cumprod_t
    )

    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(img_size: int,
                      T: int,
                      device: str,
                      model: torch.nn.Module,
                      betas: torch.Tensor,
                      sqrt_one_minus_alphas_cumprod: torch.Tensor,
                      sqrt_recip_alphas: torch.Tensor,
                      posterior_variance: torch.Tensor,
                      num_images=10,
                      num_processes=1
                      ):
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    stepsize = int(T/num_images)

    sampled_images = []
    iterator = tqdm(range(0,T)[::-1], desc="Sampling the denoising process")
    for process in range(num_processes):
        process_tracker=[]
        for i in iterator:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t, model, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                process_tracker.append(img.squeeze(0))

        sampled_images.append(process_tracker)
    show_noising_process(sampled_images)