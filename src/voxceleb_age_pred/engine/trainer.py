import math

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.voxceleb_age_pred.eval_utils import compute_metrics
from src.voxceleb_age_pred.losses import compute_total_loss


def build_optimizer(model, config):
    groups = []
    used = set()
    backbone = model.backbone
    for idx, layer in enumerate(backbone.encoder.layers):
        params = [param for param in layer.parameters() if param.requires_grad]
        if not params:
            continue
        groups.append({
            'params': params,
            'lr': config.learning_rate * (config.lr_layer_decay ** max(0, 23 - idx)),
            'weight_decay': config.weight_decay,
        })
        used.update(map(id, params))
    remaining = [param for param in model.parameters() if param.requires_grad and id(param) not in used]
    if remaining:
        groups.append({'params': remaining, 'lr': config.learning_rate, 'weight_decay': config.weight_decay})
    return torch.optim.AdamW(groups)


def _move_batch(batch, device):
    return {
        'waveform': batch['waveform'].to(device),
        'age': batch['age'].to(device),
        'coarse_bin': batch['coarse_bin'].to(device),
        'fine_bin': batch['fine_bin'].to(device),
        'sex_id': batch['sex_id'].to(device),
        'sex': batch['sex'],
        'speaker_id': batch['speaker_id'],
        'video_id': batch['video_id'],
        'clip_id': batch['clip_id'],
        'uid': batch['uid'],
    }


def apply_mixup(batch, config):
    if batch['waveform'].size(0) < 2:
        return batch
    lam = max(torch.distributions.Beta(config.mixup_alpha, config.mixup_alpha).sample().item(), 0.5)
    perm = torch.randperm(batch['waveform'].size(0), device=batch['waveform'].device)
    if config.sex_specific:
        for target in torch.unique(batch['sex_id']):
            idx = (batch['sex_id'] == target).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel(), device=batch['waveform'].device)]
    batch['waveform'] = lam * batch['waveform'] + (1.0 - lam) * batch['waveform'][perm]
    batch['age'] = lam * batch['age'] + (1.0 - lam) * batch['age'][perm]
    return batch


def _epoch_loop(model, loader, optimizer, scheduler, config, device, logger, training):
    model.train(training)
    scaler = GradScaler(enabled=config.fp16 and device.type == 'cuda') if training else None
    rows = []
    loss_meter = 0.0
    optimizer_zero = optimizer.zero_grad if optimizer is not None else (lambda: None)
    if training:
        optimizer_zero()
    iterator = tqdm(loader, leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = _move_batch(batch, device)
        if training:
            batch = apply_mixup(batch, config)
        with torch.set_grad_enabled(training):
            with autocast(enabled=config.fp16 and device.type == 'cuda'):
                outputs = model(batch['waveform'], sex_ids=batch['sex_id'])
                loss, _ = compute_total_loss(outputs, batch['age'], config)
                loss_meter += float(loss.detach().item())
            if training:
                loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if step % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
        pred_age = outputs['pred_age'].detach().cpu().tolist()
        coarse_age = outputs['coarse_age'].detach().cpu().tolist() if outputs.get('coarse_age') is not None else [''] * len(pred_age)
        for idx, pred in enumerate(pred_age):
            age_value = float(batch['age'][idx].detach().cpu().item())
            rows.append({
                'uid': batch['uid'][idx],
                'speaker_id': batch['speaker_id'][idx],
                'video_id': batch['video_id'][idx],
                'clip_id': batch['clip_id'][idx],
                'sex': batch['sex'][idx],
                'true_age': age_value,
                'pred_age': float(pred),
                'coarse_age': '' if coarse_age[idx] == '' else float(coarse_age[idx]),
                'age_decade': int(age_value // 10 * 10),
            })
    if training and len(loader) % config.gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    y_true = [row['true_age'] for row in rows]
    y_pred = [row['pred_age'] for row in rows]
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = loss_meter / max(1, len(loader))
    return metrics, rows


def train_model(model, train_loader, val_loader, config, device, logger):
    model.to(device)
    optimizer = build_optimizer(model, config)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / config.gradient_accumulation_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group['lr'] for group in optimizer.param_groups],
        epochs=config.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.warmup_ratio,
        anneal_strategy='cos',
    )
    best = None
    best_mae = float('inf')
    bad_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        train_metrics, _ = _epoch_loop(model, train_loader, optimizer, scheduler, config, device, logger, training=True)
        val_metrics, _ = _epoch_loop(model, val_loader, None, None, config, device, logger, training=False)
        logger.info('epoch=%d train=%s val=%s', epoch, train_metrics, val_metrics)
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            bad_epochs = 0
            best = {
                'epoch': epoch,
                'model_state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                'val_metrics': val_metrics,
                'config': config.to_dict(),
            }
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience:
                logger.info('Early stopping triggered at epoch %d', epoch)
                break
    return best


def predict_dataset(model, loader, config, device, logger):
    _, rows = _epoch_loop(model, loader, None, None, config, device, logger, training=False)
    if not rows:
        return {
            'uid': [], 'speaker_id': [], 'video_id': [], 'clip_id': [], 'sex': [],
            'true_age': [], 'pred_age': [], 'coarse_age': [], 'age_decade': [],
        }
    return {key: [row[key] for row in rows] for key in rows[0]}
