"""
Utils for efficient spatially varying Gaussian blur. Rough usage shown in the main function.
"""
import torch
from torch import Tensor


def precompute_gaussian_sum_els(size=11):
    center = size // 2
    x = torch.arange(-center, 0)
    y = torch.arange(-center, 1)
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Precompute x^2 + y^2
    distance_squared = x ** 2 + y ** 2

    return distance_squared


def gaussian_kernel_sum(distance_squared, sigma):
    sigma = sigma.unsqueeze(-1).unsqueeze(-1)
    return 4 * torch.exp(-distance_squared / (2 * sigma ** 2)).sum(dim=(-1, -2)) + 1


def precompute_gaussian_kernel_components(size=11):
    center = size // 2
    x = torch.arange(size) - center
    y = torch.arange(size) - center
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Precompute x^2 + y^2
    distance_squared = x ** 2 + y ** 2

    return distance_squared

def compute_gaussian_kernel_components(precomputed, sigma):
    return torch.exp(-precomputed.unsqueeze(0) / (2 * sigma ** 2))



def gaussian_kernel(stds, size=11):
    """ Takes a series of std values of length N
        and integer size corresponding to kernel side length M
        and returns a set of gaussian kernels with those stds in a (N,M,M) tensor

    Args:
        stds (Tensor): Flat list tensor containing std values.
        size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Tensor containing a unique 2D Gaussian kernel for each value in the stds input.

    """
    # 1. create input vector to the exponential function
    n = (torch.arange(0, size, device=stds.device) - (size - 1.0) / 2.0).unsqueeze(-1)
    var = 2 * (stds ** 2).unsqueeze(-1) + 1e-8  # add constant for stability

    # 2. compute gaussian values with exponential
    kernel_1d = torch.exp((-n ** 2) / var.t()).permute(1, 0)
    # 3. outer product in a batch
    kernel_2d = torch.bmm(kernel_1d.unsqueeze(2), kernel_1d.unsqueeze(1))
    # 4. normalize to unity sum
    kernel_2d /= kernel_2d.sum(dim=(-1, -2)).view(-1, 1, 1)

    return kernel_2d


def local_gaussian_blur(input, modulator, kernel_size=11):
    """Blurs image with dynamic Gaussian blur.

    Args:
        input (Tensor): The image to be blurred (C,H,W).
        modulator (Tensor): The modulating signal that determines the local value of kernel variance (H,W).
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Locally blurred version of the input image.

    """

    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    b, c, h, w = input.shape
    pad = int((kernel_size - 1) / 2)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad, pad, pad, pad), mode='constant')
    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = gaussian_kernel(modulator.flatten(), size=kernel_size).view(b, -1, kernel_size, kernel_size)
    # kernels_rgb = torch.stack(c*[kernels], 1)
    kernels_rgb = kernels.unsqueeze(1).expand(kernels.shape[0], c, *kernels.shape[1:])
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size, kernel_size))
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b, c, -1, h * w)
    x2 = kernels_rgb.view(b, c, h * w, -1).permute(0, 1, 3, 2)  # .unsqueeze(0)
    y = (x1 * x2).sum(2)
    # 5. Fold and return
    return torch.nn.functional.fold(y, (h, w), (1, 1))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    s = 500
    n = 100
    kernel_half = 20
    n_iters = 10
    kernel_size = 2 * kernel_half + 1
    # create 2d random indices
    indices = torch.randint(kernel_half, int(s/2) - kernel_half, (2, n)).to("cuda")
    # random values
    values = torch.rand((n, 1)).to("cuda")
    kernel_ids = torch.arange(-kernel_half, kernel_half + 1).to("cuda")
    kernel_ids_x, kernel_ids_y = torch.meshgrid(kernel_ids, kernel_ids)
    kernel_components_sum = precompute_gaussian_sum_els(kernel_size).to("cuda")
    kernel_components = precompute_gaussian_kernel_components(kernel_size).to("cuda")
    start = time.time()
    for i in range(n_iters):
        all_ids_ = torch.zeros((2, n, kernel_size, kernel_size)).to("cuda")
        all_ids_[0] = (indices[0].unsqueeze(-1).unsqueeze(-1) + kernel_ids_x)
        all_ids_[1] = (indices[1].unsqueeze(-1).unsqueeze(-1) + kernel_ids_y)
        all_ids = all_ids_.reshape(2, -1).unique(dim=1)  # for each of the cells, we need to compute a convolution
        depths = (all_ids[1] + all_ids[0]) / s * 5
        depths_facs = torch.sparse_coo_tensor(all_ids, depths, (s,
                                                                s), is_coalesced=True).to_dense()
        norm_facs = gaussian_kernel_sum(kernel_components_sum, depths).unsqueeze(-1)
        all_ids_ = all_ids_.to(torch.int32)
        corresponding_depths = depths_facs[all_ids_[0], all_ids_[1]]
        kernels = compute_gaussian_kernel_components(kernel_components, corresponding_depths)
        kernels_computed = kernels.unsqueeze(-1) * values.unsqueeze(1).unsqueeze(1)
        new_image = torch.sparse_coo_tensor(all_ids_.reshape(2, -1), kernels_computed.reshape(-1, 1),
                                          (s, s, 1)).coalesce()
        new_image.values().data /= norm_facs
    print("Time taken for sparse computation: ", (time.time() - start)/n_iters)
    original_image = torch.sparse_coo_tensor(indices, values.squeeze(),
                                             (s, s)).coalesce().to_dense()
    # dense computation
    start = time.time()
    for i in range(n_iters):
        original_image = torch.sparse_coo_tensor(indices, values.squeeze(),
                                                  (s, s)).coalesce().to_dense()
        new_image_ = local_gaussian_blur(original_image.unsqueeze(0), depths_facs, kernel_size=kernel_size)
    print("Time taken for dense computation: ", (time.time() - start)/n_iters)
    print(torch.abs(new_image.to_dense().squeeze() - new_image_.squeeze()).max())
    fig, axs = plt.subplots(1, 5, figsize=(30, 10))
    axs[0].imshow(original_image.cpu().numpy())
    axs[1].imshow(new_image.to_dense().cpu().numpy().squeeze())
    axs[2].imshow(depths_facs.cpu().numpy())
    axs[3].imshow(new_image_.squeeze().cpu().numpy())
    axs[4].imshow(torch.abs(new_image.to_dense().squeeze() - new_image_.squeeze()).squeeze().cpu().numpy())

    axs[0].set_title("Original Image")
    axs[1].set_title("Blurred Image")
    axs[2].set_title("noises")
    axs[3].set_title("Blurred Image Dense")

    print(original_image.to_dense().max())
    print(new_image.to_dense().max())
    print(new_image_.max())
    plt.show()





