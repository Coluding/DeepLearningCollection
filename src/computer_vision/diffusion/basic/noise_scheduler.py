import torch.nn.functional as F
import torch


def linear_beta_scheduler(
        timesteps: int, start=0.0001, end=0.02) -> torch.Tensor:
    return torch.linspace(start, end, timesteps)


def get_index_from_list(
        vals: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Tensor):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0: torch.Tensor, t: int,
                             sqrt_alphas_cumprod: torch.tensor,
                             sqrt_one_minus_alphas_cumprod: torch.tensor,
                             device: str = "cpu"):
    noise = torch.rand_like(x_0, device=device)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # mean + variance
    return sqrt_alphas_cumprod_t * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t * noise.to(device), noise.to(device)
