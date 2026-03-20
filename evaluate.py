import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.voxceleb_age_pred.config import ExperimentConfig
from src.voxceleb_age_pred.data.dataset import AudioProcessor, VoiceAgeDataset, collate_batch, prepare_records
from src.voxceleb_age_pred.data.manifest import build_split_records
from src.voxceleb_age_pred.engine.trainer import predict_dataset
from src.voxceleb_age_pred.eval_utils import apply_bias_correction, compute_report, save_metrics, save_plots, save_predictions_csv
from src.voxceleb_age_pred.models.model import VoiceAgeModel
from src.voxceleb_age_pred.utils import ensure_dir, load_json, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--save_plots', action='store_true')
    parser.add_argument('--output_dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = ExperimentConfig.from_dict(checkpoint['config'])
    output_dir = ensure_dir(args.output_dir or (config.output_dir / 'evaluation'))
    logger = setup_logger(output_dir / 'evaluate.log')

    split_records, manifest_summary = build_split_records(config, logger)
    processor = AudioProcessor(config, logger)
    prepared = {}
    for split, records in split_records.items():
        cache_path = config.output_dir / 'artifacts' / f'{split}_filter_cache.json'
        apply_pathology = split == 'train' and config.pathology_filter
        prepared[split], _ = prepare_records(records, processor, cache_path, apply_pathology, logger)

    if not prepared[args.split]:
        raise RuntimeError(f'No usable {args.split} samples were found. Check raw audio accessibility and filtering thresholds.')
    dataset = VoiceAgeDataset(prepared[args.split], config, processor, training=False)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_batch, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoiceAgeModel(config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    pred = predict_dataset(model, loader, config, device, logger)

    bias_path = config.output_dir / 'bias_correction.json'
    if config.uses_bias_correction and bias_path.exists():
        bias_payload = load_json(bias_path)
        pred['pred_age_bc'] = apply_bias_correction(pred['pred_age'], pred['sex'], bias_payload)
    report = compute_report(pred, corrected='pred_age_bc' if 'pred_age_bc' in pred else None)
    save_metrics(output_dir / f'metrics_{args.split}.json', report)
    save_predictions_csv(output_dir / f'predictions_{args.split}.csv', pred)
    if args.save_plots:
        save_plots(output_dir / 'plots', pred, uses_ldl=config.uses_ldl, uses_bc='pred_age_bc' in pred)

    print(json.dumps(report['overall'], indent=2, ensure_ascii=False))
    print('manifest_summary:', json.dumps(manifest_summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
