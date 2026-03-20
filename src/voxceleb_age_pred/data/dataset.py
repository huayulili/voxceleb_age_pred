import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

try:
    import parselmouth
except ImportError:
    parselmouth = None

from src.voxceleb_age_pred.data.manifest import AudioSource, SampleRecord
from src.voxceleb_age_pred.utils import load_json, save_json


def sex_to_id(value):
    if value == 'male':
        return 0
    if value == 'female':
        return 1
    return -1


def age_to_bin(age: torch.Tensor, delta: int, age_min: int, num_bins: int):
    bins = torch.round((age - age_min) / delta).long()
    return bins.clamp(0, num_bins - 1)


class AudioProcessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.archive_cache = {}

    def _decode_with_ffmpeg(self, path: Path = None, payload: bytes = None) -> Tuple[torch.Tensor, int]:
        source = str(path) if path is not None else 'pipe:0'
        cmd = [
            'ffmpeg',
            '-nostdin',
            '-v',
            'error',
            '-i',
            source,
            '-f',
            's16le',
            '-acodec',
            'pcm_s16le',
            '-ac',
            '1',
            '-ar',
            str(self.config.sample_rate),
            'pipe:1',
        ]
        result = subprocess.run(
            cmd,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(samples).unsqueeze(0)
        return waveform, self.config.sample_rate

    def _read_source(self, source: AudioSource) -> Tuple[torch.Tensor, int]:
        if source.store_type == 'file':
            return self._decode_with_ffmpeg(path=source.root)
        import zipfile
        with zipfile.ZipFile(source.root) as zf:
            with zf.open(source.member) as handle:
                payload = handle.read()
        return self._decode_with_ffmpeg(payload=payload)

    def load_waveform(self, source: AudioSource) -> torch.Tensor:
        waveform, sample_rate = self._read_source(source)
        waveform = waveform.mean(dim=0)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.config.sample_rate)
        return waveform.float()

    def trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() == 0:
            return waveform
        trimmed, _ = librosa.effects.trim(waveform.numpy(), top_db=30)
        return torch.from_numpy(trimmed).float()

    def estimate_snr_db(self, waveform: torch.Tensor) -> float:
        if waveform.numel() < self.config.sample_rate // 2:
            return -math.inf
        frame = max(16, int(self.config.sample_rate * 0.025))
        hop = max(8, int(self.config.sample_rate * 0.01))
        if waveform.numel() < frame:
            return -math.inf
        frames = waveform.unfold(0, frame, hop)
        energy = frames.pow(2).mean(dim=-1) + 1e-8
        signal = torch.quantile(energy, 0.9)
        noise = torch.quantile(energy, 0.1)
        return float(10.0 * torch.log10(signal / noise).item())

    def pathology_metrics(self, waveform: torch.Tensor) -> Dict[str, float]:
        if parselmouth is None:
            return {'available': False}
        sound = parselmouth.Sound(waveform.numpy(), self.config.sample_rate)
        point_process = parselmouth.praat.call(sound, 'To PointProcess (periodic, cc)', 75, 500)
        jitter = parselmouth.praat.call(point_process, 'Get jitter (local)', 0, 0, 75, 500, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], 'Get shimmer (local)', 0, 0, 75, 500, 1.3, 1.6)
        harmonicity = parselmouth.praat.call(sound, 'To Harmonicity (cc)', 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, 'Get mean', 0, 0)
        return {'available': True, 'jitter': float(jitter * 100.0), 'shimmer': float(shimmer * 100.0), 'hnr': float(hnr)}

    def inspect_record(self, record: SampleRecord, apply_pathology: bool):
        try:
            waveform = self.load_waveform(record.source)
            waveform = self.trim_silence(waveform)
            duration = waveform.numel() / self.config.sample_rate
            snr_db = self.estimate_snr_db(waveform)
            pathology = self.pathology_metrics(waveform) if apply_pathology else {'available': False}
        except Exception as exc:
            return False, {'uid': record.uid, 'error': str(exc)}

        keep = True
        reason = 'ok'
        if snr_db < self.config.snr_threshold_db:
            keep = False
            reason = 'low_snr'
        elif duration < self.config.min_duration_seconds:
            keep = False
            reason = 'short_audio'
        elif apply_pathology and pathology.get('available'):
            if pathology['jitter'] > 2.0 or pathology['shimmer'] > 6.0 or pathology['hnr'] < 10.0:
                keep = False
                reason = 'pathology_flagged'
        info = {'uid': record.uid, 'keep': keep, 'reason': reason, 'duration_seconds': duration, 'snr_db': snr_db, 'pathology': pathology}
        return keep, info

    def crop_or_pad(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        target = int(self.config.sample_rate * self.config.clip_seconds)
        if waveform.numel() >= target:
            if training:
                start = torch.randint(0, waveform.numel() - target + 1, (1,)).item()
            else:
                start = 0
            waveform = waveform[start:start + target]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target - waveform.numel()))
        return waveform

    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        mean = waveform.mean()
        std = waveform.std().clamp_min(1e-6)
        return (waveform - mean) / std

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.config.gaussian_noise_prob:
            sigma = torch.empty(1).uniform_(0.001, 0.01).item()
            waveform = waveform + torch.randn_like(waveform) * sigma
        if torch.rand(1).item() < self.config.volume_perturb_prob:
            gain = torch.empty(1).uniform_(0.8, 1.2).item()
            waveform = waveform * gain
        if torch.rand(1).item() < self.config.time_mask_prob:
            length = waveform.numel()
            frac = torch.empty(1).uniform_(0.01, 0.1).item()
            width = max(1, int(length * frac))
            start = torch.randint(0, max(1, length - width + 1), (1,)).item()
            waveform[start:start + width] = 0.0
        return waveform


def prepare_records(records: List[SampleRecord], processor: AudioProcessor, cache_path: Path, apply_pathology: bool, logger):
    if cache_path.exists():
        cache = {item['uid']: item for item in load_json(cache_path)}
    else:
        cache = {}
    kept = []
    reports = []
    for record in tqdm(records, desc=f'filter-{cache_path.stem}', leave=False):
        info = cache.get(record.uid)
        if info is None:
            _, info = processor.inspect_record(record, apply_pathology)
            cache[record.uid] = info
        reports.append(info)
        if info.get('keep', False):
            kept.append(replace(record, metadata=info))
    save_json(cache_path, list(cache.values()))
    stats = {
        'input_records': len(records),
        'kept_records': len(kept),
        'removed_records': len(records) - len(kept),
        'reason_breakdown': {},
    }
    for info in reports:
        reason = info.get('reason', 'error')
        stats['reason_breakdown'][reason] = stats['reason_breakdown'].get(reason, 0) + 1
    logger.info('Prepared %s: %s', cache_path.stem, stats)
    return kept, stats


class VoiceAgeDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[SampleRecord], config, processor: AudioProcessor, training: bool):
        self.records = records
        self.config = config
        self.processor = processor
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        waveform = self.processor.load_waveform(record.source)
        waveform = self.processor.trim_silence(waveform)
        waveform = self.processor.crop_or_pad(waveform, self.training)
        waveform = self.processor.normalize(waveform)
        if self.training:
            waveform = self.processor.augment(waveform)
        age = torch.tensor(record.speaker_age, dtype=torch.float32)
        return {
            'waveform': waveform,
            'age': age,
            'coarse_bin': age_to_bin(age.unsqueeze(0), self.config.coarse_bin_width, self.config.age_min, self.config.coarse_bins).squeeze(0),
            'fine_bin': age_to_bin(age.unsqueeze(0), self.config.fine_bin_width, self.config.age_min, self.config.fine_bins).squeeze(0),
            'sex_id': torch.tensor(sex_to_id(record.sex), dtype=torch.long),
            'sex': record.sex or 'unknown',
            'speaker_id': record.speaker_id,
            'video_id': record.video_id,
            'clip_id': record.clip_id,
            'uid': record.uid,
        }


def collate_batch(batch):
    return {
        'waveform': torch.stack([item['waveform'] for item in batch], dim=0),
        'age': torch.stack([item['age'] for item in batch], dim=0),
        'coarse_bin': torch.stack([item['coarse_bin'] for item in batch], dim=0),
        'fine_bin': torch.stack([item['fine_bin'] for item in batch], dim=0),
        'sex_id': torch.stack([item['sex_id'] for item in batch], dim=0),
        'sex': [item['sex'] for item in batch],
        'speaker_id': [item['speaker_id'] for item in batch],
        'video_id': [item['video_id'] for item in batch],
        'clip_id': [item['clip_id'] for item in batch],
        'uid': [item['uid'] for item in batch],
    }
