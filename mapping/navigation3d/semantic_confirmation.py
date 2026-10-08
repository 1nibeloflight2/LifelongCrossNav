"""Frame-local SED score used exclusively by the final found gate."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def masked_sed_top20_mean(dense: torch.Tensor, query: torch.Tensor,
                         mask: np.ndarray) -> float:
    """Match the offline score, normalizing *after* bilinear interpolation.

    Chunking the feature norm avoids two full CxHxW tensors. Neither an image
    encoder nor SAM is called here; both inputs belong to the confirmed frame.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or int(mask.sum()) < 10:
        raise ValueError("missing_or_tiny_sam_mask")
    if not isinstance(dense, torch.Tensor) or dense.ndim != 3:
        raise ValueError("missing_dense_features")
    if not isinstance(query, torch.Tensor):
        raise ValueError("missing_query_features")
    with torch.inference_mode():
        dense = dense.detach().float()
        query = query.detach().to(device=dense.device, dtype=torch.float32).reshape(-1)
        if query.numel() != dense.shape[0]:
            raise ValueError("query_feature_dimension_mismatch")
        if not torch.isfinite(dense).all() or not torch.isfinite(query).all():
            raise ValueError("nonfinite_features")
        query = query / (query.norm() + 1e-8)
        dots = torch.einsum("chw,c->hw", dense, query)
        dots = F.interpolate(dots[None, None], size=mask.shape,
                             mode="bilinear", align_corners=False)[0, 0]
        norm_sq = torch.zeros_like(dots)
        for chunk in dense.split(96, dim=0):
            expanded = F.interpolate(chunk[None], size=mask.shape,
                                     mode="bilinear", align_corners=False)[0]
            norm_sq += expanded.square().sum(dim=0)
        similarities = (dots / (norm_sq.sqrt() + 1e-8)).cpu().numpy()
    if not np.isfinite(similarities).all():
        raise ValueError("nonfinite_similarity")
    values = similarities[mask]
    k = max(1, int(math.ceil(0.20 * values.size)))
    return float(np.partition(values, values.size - k)[-k:].mean())
