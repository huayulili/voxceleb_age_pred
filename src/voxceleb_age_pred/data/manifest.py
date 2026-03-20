import csv
import json
import random
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.voxceleb_age_pred.utils import ensure_dir


@dataclass
class AudioSource:
    store_type: str
    root: str
    member: str = ''

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass
class SampleRecord:
    split: str
    speaker_age: float
    birth_year: float
    name: str
    speaker_id: str
    video_id: str
    clip_id: str
    source: AudioSource
    sex: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f'{self.split}:{self.speaker_id}:{self.video_id}:{self.clip_id}'

    def to_dict(self):
        payload = asdict(self)
        payload['source'] = self.source.to_dict()
        return payload


class AudioLocator:
    def __init__(self, root: Path, logger):
        self.root = Path(root)
        self.logger = logger
        self.zip_indexes: Dict[str, Dict[str, List[str]]] = {}
        self.stores = []
        self._discover_stores()

    def _discover_stores(self):
        candidates = [
            ('vox1_dir', self.root / 'vox1' / 'wav'),
            ('vox2_dir_aac', self.root / 'vox2' / 'aac'),
            ('vox2_dir_wav', self.root / 'vox2' / 'wav'),
            ('vox1_test_zip', self.root / 'vox1' / 'vox1_test_wav.zip'),
            ('vox1_dev_zip', self.root / 'vox1' / 'vox1_dev_wav.zip'),
            ('vox2_dev_zip', self.root / 'vox2' / 'vox2_dev_aac.zip'),
        ]
        for name, path in candidates:
            if path.exists():
                self.stores.append((name, path))
        split_part_candidates = [
            self.root / 'vox1' / 'vox1_dev_wav_partaa',
            self.root / 'vox2' / 'vox2_dev_aac_partaa',
        ]
        unsupported = [str(path) for path in split_part_candidates if path.exists()]
        if unsupported:
            self.logger.warning('Detected split archives without merged zip support: %s', unsupported)

    def _zip_index(self, archive_path: Path) -> Dict[str, List[str]]:
        cache_key = str(archive_path)
        if cache_key in self.zip_indexes:
            return self.zip_indexes[cache_key]
        index: Dict[str, List[str]] = {}
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if member.endswith('/'):
                    continue
                parts = member.split('/')
                if len(parts) < 4:
                    continue
                prefix = '/'.join(parts[:3])
                index.setdefault(prefix, []).append(member)
        self.zip_indexes[cache_key] = index
        return index

    def list_sources(self, speaker_id: str, video_id: str) -> List[AudioSource]:
        collected: List[AudioSource] = []
        for store_type, path in self.stores:
            if path.is_dir():
                target = path / speaker_id / video_id
                if target.exists():
                    for item in sorted(target.iterdir()):
                        if item.suffix.lower() in {'.wav', '.aac', '.m4a', '.flac'}:
                            collected.append(AudioSource(store_type='file', root=str(item)))
                continue
            prefix_root = 'wav' if 'vox1' in store_type else 'aac'
            prefix = f'{prefix_root}/{speaker_id}/{video_id}'
            for member in self._zip_index(path).get(prefix, []):
                collected.append(AudioSource(store_type='zip', root=str(path), member=member))
        return collected


def _normalize_sex(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {'m', 'male'}:
        return 'male'
    if text in {'f', 'female'}:
        return 'female'
    return None


def _read_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    text = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    rows = [line for line in text if line.strip()]
    if not rows:
        return [], []
    delimiter = '	' if '	' in rows[0] else ','
    reader = csv.reader(rows, delimiter=delimiter)
    parsed = list(reader)
    header = [cell.strip().replace('﻿', '') for cell in parsed[0]]
    return header, parsed[1:]


def load_gender_map(dataset_root: Path, voxceleb_root: Path) -> Dict[str, str]:
    mapping = {}
    meta_paths = [
        dataset_root / 'vox1_meta.csv',
        dataset_root / 'vox2_meta.csv',
        voxceleb_root / 'vox1' / 'vox1_meta.csv',
        voxceleb_root / 'vox2' / 'vox2_meta.csv',
    ]
    for path in meta_paths:
        if not path.exists():
            continue
        header, rows = _read_table(path)
        lowered = [item.lower().strip() for item in header]
        id_idx = next((i for i, item in enumerate(lowered) if 'voxceleb' in item and 'id' in item), None)
        gender_idx = next((i for i, item in enumerate(lowered) if 'gender' in item), None)
        if id_idx is None or gender_idx is None:
            continue
        for row in rows:
            if id_idx >= len(row) or gender_idx >= len(row):
                continue
            speaker_id = row[id_idx].strip()
            sex = _normalize_sex(row[gender_idx])
            if speaker_id and sex:
                mapping[speaker_id] = sex
    return mapping


def load_age_rows(path: Path) -> List[dict]:
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as handle:
        return list(csv.DictReader(handle))


def split_train_val_rows(train_rows: List[dict], fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    speakers = sorted({row['VoxCeleb_ID'] for row in train_rows})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    val_count = max(1, int(len(speakers) * fraction))
    val_speakers = set(speakers[:val_count])
    train_split = [row for row in train_rows if row['VoxCeleb_ID'] not in val_speakers]
    val_split = [row for row in train_rows if row['VoxCeleb_ID'] in val_speakers]
    return train_split, val_split


def expand_rows(rows: List[dict], split: str, locator: AudioLocator, gender_map: Dict[str, str]) -> Tuple[List[SampleRecord], dict]:
    records: List[SampleRecord] = []
    missing = []
    for row in rows:
        sources = locator.list_sources(row['VoxCeleb_ID'], row['video_id'])
        if not sources:
            missing.append({'speaker_id': row['VoxCeleb_ID'], 'video_id': row['video_id'], 'name': row['Name']})
            continue
        for source in sources:
            clip_id = Path(source.member or source.root).stem
            records.append(SampleRecord(
                split=split,
                speaker_age=float(row['speaker_age']),
                birth_year=float(row['birth_year']),
                name=row['Name'],
                speaker_id=row['VoxCeleb_ID'],
                video_id=row['video_id'],
                clip_id=clip_id,
                source=source,
                sex=gender_map.get(row['VoxCeleb_ID']),
            ))
    summary = {
        'rows': len(rows),
        'expanded_clips': len(records),
        'missing_rows': len(missing),
        'missing_examples': missing[:20],
    }
    return records, summary


def build_split_records(config, logger):
    dataset_root = Path(config.dataset_root)
    age_train = load_age_rows(dataset_root / 'age-train.txt')
    age_test = load_age_rows(dataset_root / 'age-test.txt')
    train_rows, val_rows = split_train_val_rows(age_train, config.train_val_speaker_fraction, config.seed)
    gender_map = load_gender_map(dataset_root, Path(config.voxceleb_root))
    locator = AudioLocator(Path(config.voxceleb_root), logger)
    manifests = {}
    summaries = {'gender_map_entries': len(gender_map)}
    for split, rows in [('train', train_rows), ('val', val_rows), ('test', age_test)]:
        manifests[split], summaries[split] = expand_rows(rows, split, locator, gender_map)
        logger.info('%s summary: %s', split, json.dumps(summaries[split], ensure_ascii=False))
    return manifests, summaries
