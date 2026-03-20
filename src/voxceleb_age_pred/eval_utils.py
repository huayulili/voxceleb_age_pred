import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.voxceleb_age_pred.utils import ensure_dir, save_json


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    metrics = {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'median_ae': float(np.median(np.abs(y_true - y_pred))),
        'within_1': float(np.mean(np.abs(y_true - y_pred) <= 1.0)),
        'within_3': float(np.mean(np.abs(y_true - y_pred) <= 3.0)),
        'within_5': float(np.mean(np.abs(y_true - y_pred) <= 5.0)),
    }
    metrics['pearson_r'] = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else 0.0
    return metrics


def _group_report(payload, key, pred_key):
    groups = {}
    for idx, value in enumerate(payload[key]):
        groups.setdefault(value, {'true': [], 'pred': []})
        groups[value]['true'].append(payload['true_age'][idx])
        groups[value]['pred'].append(payload[pred_key][idx])
    return {str(group): compute_metrics(item['true'], item['pred']) for group, item in groups.items() if item['true']}


def compute_report(payload, corrected=None):
    report = {'overall': compute_metrics(payload['true_age'], payload['pred_age'])}
    report['by_age_decade'] = _group_report(payload, 'age_decade', 'pred_age')
    report['by_sex'] = _group_report(payload, 'sex', 'pred_age')
    if corrected:
        report['overall_corrected'] = compute_metrics(payload['true_age'], payload[corrected])
        report['by_age_decade_corrected'] = _group_report(payload, 'age_decade', corrected)
        report['by_sex_corrected'] = _group_report(payload, 'sex', corrected)
    return report


def fit_bias_correction(payload):
    params = {'global': {}}
    x = np.asarray(payload['true_age'], dtype=np.float32).reshape(-1, 1)
    y = np.asarray(payload['pred_age'], dtype=np.float32)
    reg = LinearRegression().fit(x, y)
    params['global'] = {'alpha': float(reg.coef_[0]), 'beta': float(reg.intercept_)}
    by_sex = {}
    for sex in sorted(set(payload['sex'])):
        idx = [i for i, value in enumerate(payload['sex']) if value == sex]
        if len(idx) < 2:
            continue
        x_s = x[idx]
        y_s = y[idx]
        reg_s = LinearRegression().fit(x_s, y_s)
        by_sex[sex] = {'alpha': float(reg_s.coef_[0]), 'beta': float(reg_s.intercept_)}
    params['by_sex'] = by_sex
    return params


def apply_bias_correction(pred_age, sex, params):
    corrected = []
    global_params = params['global']
    for value, sex_value in zip(pred_age, sex):
        item = params.get('by_sex', {}).get(sex_value, global_params)
        alpha = item['alpha'] if abs(item['alpha']) > 1e-6 else 1.0
        corrected.append(float((value - item['beta']) / alpha))
    return corrected


def save_predictions_csv(path: Path, payload):
    ensure_dir(Path(path).parent)
    keys = list(payload.keys())
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in zip(*[payload[key] for key in keys]):
            writer.writerow(row)


def save_metrics(path: Path, report):
    save_json(path, report)


def save_plots(output_dir: Path, payload, uses_ldl: bool, uses_bc: bool):
    output_dir = ensure_dir(output_dir)
    true_age = np.asarray(payload['true_age'])
    pred_age = np.asarray(payload['pred_age'])
    plt.figure(figsize=(6, 6))
    plt.scatter(true_age, pred_age, s=8, alpha=0.5)
    lims = [min(true_age.min(), pred_age.min()), max(true_age.max(), pred_age.max())]
    plt.plot(lims, lims, 'k--')
    coef = np.polyfit(true_age, pred_age, 1)
    xline = np.linspace(*lims, 100)
    plt.plot(xline, coef[0] * xline + coef[1], color='tomato')
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.tight_layout()
    plt.savefig(output_dir / '01_scatter.png', dpi=200)
    plt.close()

    diff = pred_age - true_age
    mean = (pred_age + true_age) / 2.0
    sd = diff.std()
    plt.figure(figsize=(6, 5))
    plt.scatter(mean, diff, s=8, alpha=0.5)
    plt.axhline(diff.mean(), color='black')
    plt.axhline(diff.mean() + 1.96 * sd, color='red', linestyle='--')
    plt.axhline(diff.mean() - 1.96 * sd, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / '02_bland_altman.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(diff, bins=40, color='steelblue', alpha=0.8)
    mae = np.mean(np.abs(diff))
    plt.axvline(mae, color='darkorange', linestyle='--')
    plt.axvline(-mae, color='darkorange', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / '03_gap_hist.png', dpi=200)
    plt.close()

    decades = np.asarray(payload['age_decade'])
    groups = sorted(set(decades.tolist()))
    maes = []
    counts = []
    for group in groups:
        idx = decades == group
        maes.append(np.mean(np.abs(pred_age[idx] - true_age[idx])))
        counts.append(int(idx.sum()))
    plt.figure(figsize=(7, 4))
    bars = plt.bar([str(x) for x in groups], maes, color='seagreen')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / '04_mae_by_age.png', dpi=200)
    plt.close()

    if uses_ldl and payload.get('coarse_age'):
        coarse = np.asarray(payload['coarse_age'])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(true_age, coarse, s=8, alpha=0.5)
        axes[0].set_title('Coarse')
        axes[1].scatter(true_age, pred_age, s=8, alpha=0.5)
        axes[1].set_title('Refined')
        for axis in axes:
            axis.plot(lims, lims, 'k--')
        fig.tight_layout()
        fig.savefig(output_dir / '05_coarse_vs_refined.png', dpi=200)
        plt.close(fig)

    if uses_bc and payload.get('pred_age_bc'):
        corrected = np.asarray(payload['pred_age_bc'])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(true_age, pred_age - true_age, s=8, alpha=0.5)
        axes[0].set_title('Before correction')
        axes[1].scatter(true_age, corrected - true_age, s=8, alpha=0.5)
        axes[1].set_title('After correction')
        fig.tight_layout()
        fig.savefig(output_dir / '06_bias_correction.png', dpi=200)
        plt.close(fig)


def prediction_payload_from_batches(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}
