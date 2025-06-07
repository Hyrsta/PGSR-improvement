import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def error_map_colors(gaussians, colormap_name='hot') -> torch.Tensor:
    """
    Generates a heatmap based on the scale of the gaussian in the normal direction.
    """
    # 1. Compute scale in the direction of the normal
    smallest_scales = gaussians.get_scaling.min(dim=-1)[0]

    if torch.isnan(smallest_scales).any():
        smallest_scales = torch.where(torch.isnan(smallest_scales), torch.zeros_like(smallest_scales), smallest_scales)

    print(f"gini coefficient before log scaling: {gini_coefficient(smallest_scales.cpu().numpy())}")

    # 2. Normalize the scales to [0, 1]
    normalized_scales = log_normalize_scales(smallest_scales)  # Shape: (N,)
    print(f"Min normalized scale: {normalized_scales.min():.4f}, Max normalized scale: {normalized_scales.max():.4f}")
    
    print(f"gini coefficient after log scaling: {gini_coefficient(normalized_scales.cpu().numpy())}")

    # 3. Adjust the normalized scales to only show the specified percentage range
    adjusted_normalized_scales = percentile_normalize_scales(normalized_scales, lower=50, upper=100)
    print(f"After adjustment - Min: {adjusted_normalized_scales.min():.4f}, Max: {adjusted_normalized_scales.max():.4f}")

    print(f"gini coefficient after log scaling and clipping: {gini_coefficient(adjusted_normalized_scales.cpu().numpy())}")

    # 4. Map the adjusted scales to RGB colors
    custom_colors = map_scales_to_colors(adjusted_normalized_scales, colormap_name=colormap_name)  # Shape: (N, 3)

    # 5. Convert from BGR to RGB
    custom_colors = custom_colors[:, [2, 1, 0]]  # (N, 3): Swap channels to get RGB
    return custom_colors

def gini_coefficient(x):
    """
    Compute the Gini coefficient of a one-dimensional array x.
    The Gini coefficient measures statistical dispersion, ranging from 0 (perfect equality)
    to 1 (maximum inequality). This implementation handles non-negative values.
    """
    # Convert to numpy array
    array = np.array(x, dtype=np.float64)
    
    # If there are negative values, shift the entire array so that minimum becomes zero
    if np.amin(array) < 0:
        array -= np.min(array)
    
    # Avoid division by zero if all values are zero
    if np.sum(array) == 0:
        return 0.0

    # Sort the array in ascending order
    sorted_array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    
    # Compute Gini using the formula:
    # G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n
    gini = (2 * np.sum(index * sorted_array) / (n * np.sum(sorted_array))) - ((n + 1) / n)
    return gini

def map_scales_to_colors(normalized_scales: torch.Tensor, colormap_name: str = 'hot') -> torch.Tensor:
    """
    Maps normalized scale values to RGB colors using a specified colormap.
    """
    scales_cpu = normalized_scales.detach().cpu().numpy() # Convert tensor to CPU and NumPy for Matplotlib processing
    cmap = cm.get_cmap(colormap_name) # Get the colormap
    colors_rgba = cmap(scales_cpu)  # Map normalized scales to RGBA colors. Shape: (N, 4)
    colors_rgb = torch.tensor(colors_rgba[:, :3], dtype=torch.float32, device=normalized_scales.device) # Convert to RGB by discarding the alpha channel and to torch.Tensor
    return colors_rgb

def plot_scale_distribution(normalized_scales, title="Normalized Scales Distribution"):
    scales_np = normalized_scales.detach().cpu().numpy()
    plt.figure(figsize=(8,6))
    plt.hist(scales_np, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Normalized Scale')
    plt.ylabel('Frequency')
    plt.show()

def percentile_normalize_scales(scales: torch.Tensor, lower=50, upper=100) -> torch.Tensor:
    """
    This method uses the 50th and 100th percentiles to define the normalization range, reducing the impact of outliers and enhancing the contrast for the majority of values.
    """
    lower_val = torch.quantile(scales, lower / 100.0)
    upper_val = torch.quantile(scales, upper / 100.0)
    normalized_scales = (scales - lower_val) / (upper_val - lower_val)
    normalized_scales = torch.clamp(normalized_scales, 0.0, 1.0)
    return normalized_scales

def log_normalize_scales(scales: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    """
    Applies logarithmic scaling to scale values before normalization.
    If the scale values span multiple orders of magnitude, applying a logarithmic transformation can help distribute the values more evenly.
    """
    log_scales = torch.log(scales + epsilon)
    min_log = log_scales.min()
    max_log = log_scales.max()
    normalized_scales = (log_scales - min_log) / (max_log - min_log)
    return normalized_scales

def normalize_scales(scales: torch.Tensor) -> torch.Tensor:
    min_scale = scales.min()
    max_scale = scales.max()
    normalized_scales = (scales - min_scale) / (max_scale - min_scale)
    return normalized_scales

def validate_override_color(custom_colors, gaussian_model):
    """
    Validates the override_color tensor.
    """
    assert isinstance(custom_colors, torch.Tensor), "override_color must be a torch.Tensor."
    assert custom_colors.dtype == torch.float32, "override_color must be of dtype torch.float32."
    assert custom_colors.device == gaussian_model.get_xyz.device, "override_color must be on the same device as GaussianModel."
    assert custom_colors.shape[1] == 3, "override_color must have 3 color channels (RGB)."
    assert custom_colors.shape[0] == gaussian_model.get_xyz.shape[0], "override_color must match the number of Gaussians."
    assert torch.all(custom_colors >= 0.0) and torch.all(custom_colors <= 1.0), "Color values must be in [0.0, 1.0]."