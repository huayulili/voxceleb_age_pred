from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BaselineHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, global_feat, local_feat):
        pred_age = self.mlp(global_feat).squeeze(-1)
        return {
            'pred_age': pred_age,
            'coarse_logits': None,
            'coarse_prob': None,
            'coarse_age': None,
            'fine_logits': None,
            'fine_prob': None,
            'ordinal_feat': global_feat,
        }


class LDLHead(nn.Module):
    def __init__(self, hidden_size: int, coarse_bins: int, fine_bins: int, age_min: int, coarse_step: int, fine_step: int, no_stage2: bool):
        super().__init__()
        self.age_min = age_min
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.no_stage2 = no_stage2
        self.coarse_bins = coarse_bins
        self.fine_bins = fine_bins
        self.coarse_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, coarse_bins),
        )
        self.age_queries = nn.Parameter(torch.randn(coarse_bins, 256))
        self.local_proj = nn.Linear(hidden_size, 256)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=4, dropout=0.1, batch_first=True)
        self.coarse_proj = nn.Linear(coarse_bins, 256)
        self.fine_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, fine_bins),
        )

    def _expected_age(self, prob, step):
        centers = torch.arange(prob.size(-1), device=prob.device, dtype=prob.dtype) * step + self.age_min
        return (prob * centers).sum(dim=-1)

    def forward(self, global_feat, local_feat):
        coarse_logits = self.coarse_head(global_feat)
        coarse_prob = coarse_logits.softmax(dim=-1)
        coarse_age = self._expected_age(coarse_prob, self.coarse_step)
        if self.no_stage2:
            return {
                'pred_age': coarse_age,
                'coarse_logits': coarse_logits,
                'coarse_prob': coarse_prob,
                'coarse_age': coarse_age,
                'fine_logits': None,
                'fine_prob': None,
                'ordinal_feat': global_feat,
            }
        coarse_index = coarse_prob.argmax(dim=-1)
        query = self.age_queries[coarse_index].unsqueeze(1)
        local_proj = self.local_proj(local_feat)
        refined_feat, _ = self.cross_attn(query, local_proj, local_proj, need_weights=False)
        refined_feat = refined_feat.squeeze(1)
        coarse_feat = self.coarse_proj(coarse_prob)
        fusion = torch.cat([refined_feat, coarse_feat], dim=-1)
        fine_logits = self.fine_head(fusion)
        fine_prob = fine_logits.softmax(dim=-1)
        pred_age = self._expected_age(fine_prob, self.fine_step)
        return {
            'pred_age': pred_age,
            'coarse_logits': coarse_logits,
            'coarse_prob': coarse_prob,
            'coarse_age': coarse_age,
            'fine_logits': fine_logits,
            'fine_prob': fine_prob,
            'ordinal_feat': refined_feat,
        }


class VoiceAgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.wavlm_name)
        self.layer_weights = nn.Parameter(torch.zeros(25))
        self._freeze_layers()
        if config.uses_ldl:
            head_factory = lambda: LDLHead(
                hidden_size=config.hidden_size,
                coarse_bins=config.coarse_bins,
                fine_bins=config.fine_bins,
                age_min=config.age_min,
                coarse_step=config.coarse_bin_width,
                fine_step=config.fine_bin_width,
                no_stage2=config.no_stage2,
            )
        else:
            head_factory = lambda: BaselineHead(config.hidden_size)
        self.shared_head = head_factory()
        self.male_head = head_factory() if config.sex_specific else None
        self.female_head = head_factory() if config.sex_specific else None

    def _freeze_layers(self):
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False
        for idx, layer in enumerate(self.backbone.encoder.layers):
            if idx <= 15:
                for param in layer.parameters():
                    param.requires_grad = False

    def _weighted_hidden(self, hidden_states):
        stacked = torch.stack(hidden_states, dim=0)
        weights = torch.softmax(self.layer_weights[: stacked.size(0)], dim=0).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def _route_head(self, global_feat, local_feat, sex_ids):
        if not self.config.sex_specific or sex_ids is None:
            return self.shared_head(global_feat, local_feat)
        outputs: Dict[str, Optional[torch.Tensor]] = {}
        groups = [
            (sex_ids == 0, self.male_head),
            (sex_ids == 1, self.female_head),
            (sex_ids < 0, self.shared_head),
        ]
        batch_size = global_feat.size(0)
        for mask, head in groups:
            if head is None or not mask.any():
                continue
            part = head(global_feat[mask], local_feat[mask])
            for key, value in part.items():
                if value is None:
                    outputs.setdefault(key, None)
                    continue
                if key not in outputs or outputs[key] is None:
                    shape = (batch_size,) + value.shape[1:]
                    outputs[key] = torch.zeros(shape, dtype=value.dtype, device=value.device)
                outputs[key][mask] = value
        return outputs

    def forward(self, input_values, attention_mask=None, sex_ids=None):
        backbone_out = self.backbone(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        fused = self._weighted_hidden(backbone_out.hidden_states)
        global_feat = fused.mean(dim=1)
        local_feat = fused
        outputs = self._route_head(global_feat, local_feat, sex_ids)
        outputs['global_feat'] = global_feat
        outputs['local_feat'] = local_feat
        return outputs
