from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    mode: str = 'ldl_bc'
    sex_specific: bool = False
    no_stage2: bool = False
    lambda_ordinal: float = 0.1
    pathology_filter: bool = True
    output_dir: Path = Path('./outputs/default')
    dataset_root: Path = Path('/home/bld/data1_disk/daavee/dataset/voice_dataset/voxceleb_enrichment_age_gender/dataset')
    voxceleb_root: Path = Path('/data1/daavee/dataset/voice_dataset/voxceleb')
    sample_rate: int = 16000
    clip_seconds: float = 6.0
    snr_threshold_db: float = 10.0
    min_duration_seconds: float = 1.0
    train_val_speaker_fraction: float = 0.1
    coarse_bin_width: int = 5
    fine_bin_width: int = 1
    age_min: int = 10
    age_max: int = 90
    batch_size: int = 16
    max_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_layer_decay: float = 0.75
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 4
    mixup_alpha: float = 0.2
    gaussian_noise_prob: float = 0.5
    volume_perturb_prob: float = 0.5
    time_mask_prob: float = 0.3
    seed: int = 42
    wavlm_name: str = 'microsoft/wavlm-large'
    hidden_size: int = 1024
    save_plots: bool = False

    @property
    def uses_ldl(self) -> bool:
        return self.mode.startswith('ldl')

    @property
    def uses_bias_correction(self) -> bool:
        return self.mode.endswith('_bc')

    @property
    def coarse_bins(self) -> int:
        return ((self.age_max - self.age_min) // self.coarse_bin_width) + 1

    @property
    def fine_bins(self) -> int:
        return ((self.age_max - self.age_min) // self.fine_bin_width) + 1

    @classmethod
    def from_namespace(cls, args):
        data = vars(args).copy()
        data['output_dir'] = Path(data['output_dir'])
        data['dataset_root'] = Path(data['dataset_root'])
        data['voxceleb_root'] = Path(data['voxceleb_root'])
        return cls(**data)

    @classmethod
    def from_dict(cls, data):
        payload = data.copy()
        payload['output_dir'] = Path(payload['output_dir'])
        payload['dataset_root'] = Path(payload['dataset_root'])
        payload['voxceleb_root'] = Path(payload['voxceleb_root'])
        return cls(**payload)

    def to_dict(self):
        payload = asdict(self)
        payload['output_dir'] = str(self.output_dir)
        payload['dataset_root'] = str(self.dataset_root)
        payload['voxceleb_root'] = str(self.voxceleb_root)
        return payload
