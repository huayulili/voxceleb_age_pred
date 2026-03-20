import torch
import torch.nn.functional as F

from src.voxceleb_age_pred.data.dataset import age_to_bin


def ldl_loss(logits, prob, true_age, delta, age_min, num_bins):
    target = age_to_bin(true_age, delta, age_min, num_bins)
    ce = F.cross_entropy(logits, target, label_smoothing=0.05)
    centers = torch.arange(num_bins, device=true_age.device, dtype=true_age.dtype) * delta + age_min
    expected_age = (prob * centers).sum(dim=-1)
    mean_penalty = 0.5 * ((true_age - expected_age) ** 2).mean()
    variance_penalty = 0.5 * ((prob * (centers.unsqueeze(0) - expected_age.unsqueeze(-1)) ** 2).sum(dim=-1)).mean()
    return ce + mean_penalty + variance_penalty


def ordinal_alignment_loss(features, ages):
    if features.size(0) < 2:
        return torch.zeros((), device=features.device)
    feat_dist = torch.pdist(features, p=2)
    age_dist = torch.pdist(ages.unsqueeze(-1), p=2)
    feat_centered = feat_dist - feat_dist.mean()
    age_centered = age_dist - age_dist.mean()
    denom = feat_centered.std().clamp_min(1e-6) * age_centered.std().clamp_min(1e-6)
    corr = (feat_centered * age_centered).mean() / denom
    return 1.0 - corr


def compute_total_loss(outputs, ages, config):
    if not config.uses_ldl:
        loss = torch.abs(outputs['pred_age'] - ages).mean()
        return loss, {'mae': float(loss.detach().item())}
    coarse_loss = ldl_loss(outputs['coarse_logits'], outputs['coarse_prob'], ages, config.coarse_bin_width, config.age_min, config.coarse_bins)
    if config.no_stage2:
        fine_loss = torch.zeros((), device=ages.device)
    else:
        fine_loss = ldl_loss(outputs['fine_logits'], outputs['fine_prob'], ages, config.fine_bin_width, config.age_min, config.fine_bins)
    ordinal = ordinal_alignment_loss(outputs['ordinal_feat'], ages)
    total = coarse_loss + fine_loss + config.lambda_ordinal * ordinal
    return total, {
        'coarse_loss': float(coarse_loss.detach().item()),
        'fine_loss': float(fine_loss.detach().item()),
        'ordinal_loss': float(ordinal.detach().item()),
    }
