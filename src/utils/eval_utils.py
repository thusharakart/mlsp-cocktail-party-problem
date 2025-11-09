import torch
from itertools import permutations


def si_snr_loss(estimated, target, eps=1e-8):
    """
    Calculate Scale-Invariant SNR per sample and source

    Args:
        estimated: [batch, n_src, time]
        target: [batch, n_src, time]

    Returns:
        si_snr: [batch, n_src] - SI-SNR in dB for each sample and source
    """
    # Truncate to same length
    min_len = min(estimated.shape[-1], target.shape[-1])
    estimated = estimated[..., :min_len]
    target = target[..., :min_len]

    # normalize
    target = target - target.mean(dim=-1, keepdim=True)
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)

    # compute SI-SNR
    s_target = (torch.sum(estimated * target, dim=-1, keepdim=True) /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target

    e_noise = estimated - s_target

    si_snr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )

    return si_snr


def pit_si_snr_loss(estimated, target, eps=1e-8):
    """
    Permutation Invariant Training (PIT) with SI-SNR loss
    
    Args:
        estimated: [batch, n_src, time] - estimated sources
        target: [batch, n_src, time] - target sources
        eps: small constant for numerical stability
    
    Returns:
        loss: [batch] - best SI-SNR for each batch sample (higher is better)
    """
    batch_size, n_src, _ = estimated.shape
    
    # get all possible permutations of source order
    perms = list(permutations(range(n_src)))
    
    # gompute SI-SNR for each permutation
    losses = []
    for perm in perms:
        # reorder target according to this permutation
        target_perm = target[:, perm, :]
        
        # compute SI-SNR for this permutation: [batch, n_src]
        si_snr = si_snr_loss(estimated, target_perm, eps)
        
        # average across sources: [batch]
        si_snr_mean = si_snr.mean(dim=1)
        losses.append(si_snr_mean)
    
    # stack all permutations: [batch, n_perms]
    losses = torch.stack(losses, dim=1)
    
    # take the best (maximum) SI-SNR across permutations
    best_si_snr, _ = torch.max(losses, dim=1)  # [batch]
    
    return best_si_snr


# Truncate to same length
def truncate_tensors(estimated, target):
    min_len = min(estimated.shape[-1], target.shape[-1])
    estimated = estimated[..., :min_len]
    target = target[..., :min_len]
    return estimated, target