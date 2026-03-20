import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.voxceleb_age_pred.config import ExperimentConfig
from src.voxceleb_age_pred.data.dataset import AudioProcessor, VoiceAgeDataset, collate_batch, prepare_records
from src.voxceleb_age_pred.data.manifest import build_split_records
from src.voxceleb_age_pred.engine.trainer import predict_dataset, train_model
from src.voxceleb_age_pred.eval_utils import (
    apply_bias_correction,
    compute_report,
    fit_bias_correction,
    save_metrics,
    save_plots,
    save_predictions_csv,
)
from src.voxceleb_age_pred.models.model import VoiceAgeModel
from src.voxceleb_age_pred.utils import ensure_dir, save_json, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train voice age prediction model.')
    parser.add_argument('--mode', default='ldl_bc', choices=['baseline', 'baseline_bc', 'ldl', 'ldl_bc'])
    parser.add_argument('--sex_specific', action='store_true')
    parser.add_argument('--no_stage2', action='store_true')
    parser.add_argument('--lambda_ordinal', type=float, default=0.1)
    parser.add_argument('--pathology_filter', dest='pathology_filter', action='store_true')
    parser.add_argument('--no_pathology_filter', dest='pathology_filter', action='store_false')
    parser.set_defaults(pathology_filter=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_root', default='/home/bld/data1_disk/daavee/dataset/voice_dataset/voxceleb_enrichment_age_gender/dataset')
    parser.add_argument('--voxceleb_root', default='/data1/daavee/dataset/voice_dataset/voxceleb')
    parser.add_argument('--wavlm_name', default='microsoft/wavlm-large')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr_layer_decay', type=float, default=0.75)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stopping_patience', type=int, default=8)
    parser.add_argument('--save_plots', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_namespace(args)
    set_seed(config.seed)
    ensure_dir(config.output_dir)
    ensure_dir(config.output_dir / 'artifacts')
    logger = setup_logger(config.output_dir / 'train.log')
    save_json(config.output_dir / 'run_config.json', config.to_dict())
    logger.info('Building manifests from %s and %s', config.dataset_root, config.voxceleb_root)
    split_records, manifest_summary = build_split_records(config, logger)
    save_json(config.output_dir / 'artifacts' / 'manifest_summary.json', manifest_summary)

    processor = AudioProcessor(config, logger)
    prepared = {}
    filter_stats = {}
    for split, records in split_records.items():
        cache_path = config.output_dir / 'artifacts' / f'{split}_filter_cache.json'
        apply_pathology = split == 'train' and config.pathology_filter
        prepared[split], filter_stats[split] = prepare_records(records, processor, cache_path, apply_pathology, logger)
    save_json(config.output_dir / 'artifacts' / 'filter_stats.json', filter_stats)

    if config.sex_specific and not any(record.sex in {'male', 'female'} for record in prepared['train']):
        logger.warning('Sex metadata unavailable; falling back to a shared head.')
        config.sex_specific = False
        save_json(config.output_dir / 'run_config.json', config.to_dict())

    for split in ('train', 'val', 'test'):
        if not prepared[split]:
            raise RuntimeError(f'No usable {split} samples were found. Check raw audio accessibility and filtering thresholds.')

    train_dataset = VoiceAgeDataset(prepared['train'], config, processor, training=True)
    train_eval_dataset = VoiceAgeDataset(prepared['train'], config, processor, training=False)
    val_dataset = VoiceAgeDataset(prepared['val'], config, processor, training=False)
    test_dataset = VoiceAgeDataset(prepared['test'], config, processor, training=False)
    loaders = {
        'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_batch, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_batch, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_batch, pin_memory=True),
    }
    eval_loaders = {
        'train': DataLoader(train_eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_batch, pin_memory=True),
        'val': loaders['val'],
        'test': loaders['test'],
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoiceAgeModel(config)
    logger.info('Training on device=%s, ldl=%s, bias_correction=%s', device, config.uses_ldl, config.uses_bias_correction)
    best_checkpoint = train_model(model, loaders['train'], loaders['val'], config, device, logger)
    checkpoint_path = config.output_dir / 'best_model.pt'
    torch.save(best_checkpoint, checkpoint_path)
    logger.info('Saved best checkpoint to %s', checkpoint_path)

    best_model = VoiceAgeModel(config)
    best_model.load_state_dict(best_checkpoint['model_state'])
    best_model.to(device)

    train_pred = predict_dataset(best_model, eval_loaders['train'], config, device, logger)
    val_pred = predict_dataset(best_model, eval_loaders['val'], config, device, logger)
    test_pred = predict_dataset(best_model, eval_loaders['test'], config, device, logger)

    bias_payload = None
    if config.uses_bias_correction:
        bias_payload = fit_bias_correction(train_pred)
        save_json(config.output_dir / 'bias_correction.json', bias_payload)
        val_pred['pred_age_bc'] = apply_bias_correction(val_pred['pred_age'], val_pred['sex'], bias_payload)
        test_pred['pred_age_bc'] = apply_bias_correction(test_pred['pred_age'], test_pred['sex'], bias_payload)

    report = {
        'train': compute_report(train_pred),
        'val': compute_report(val_pred, corrected='pred_age_bc' if bias_payload else None),
        'test': compute_report(test_pred, corrected='pred_age_bc' if bias_payload else None),
    }
    save_metrics(config.output_dir / 'metrics.json', report)
    save_predictions_csv(config.output_dir / 'predictions_val.csv', val_pred)
    save_predictions_csv(config.output_dir / 'predictions_test.csv', test_pred)

    if config.save_plots:
        save_plots(config.output_dir / 'plots' / 'val', val_pred, uses_ldl=config.uses_ldl, uses_bc=bool(bias_payload))
        save_plots(config.output_dir / 'plots' / 'test', test_pred, uses_ldl=config.uses_ldl, uses_bc=bool(bias_payload))

    logger.info('Training complete. Test MAE: %.4f', report['test']['overall']['mae'])
    print(json.dumps(report['test']['overall'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
