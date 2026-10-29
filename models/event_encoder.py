import torch
from torch import nn
from .event_data import STATIC_NAMES, REGIONS, RBP_FIELDS

# 정규화된 event 배열을 tensor로 변환하는 함수
def to_torch_batch(batch, device='cpu'):
    if not batch.get('normalized', False):
        raise ValueError('Normalize with training-only statistics before encoding')

    result = {name: torch.as_tensor(batch[name], device=device) for name in ('static_values', 'static_observed', 'rbp_values', 'rbp_observed', 'query_valid')}
    result['signature'] = batch['signature']

    return result

class EventBatchCollator:

    # Event store와 normalizer를 보관하는 함수
    def __init__(self, store, normalizer):
        self.store, self.normalizer = (store, normalizer)

    # Event ID 목록을 정규화된 tensor batch로 만드는 함수
    def __call__(self, event_ids):
        batch = self.normalizer.transform(self.store.batch(self.store.indices(event_ids)))
        result = to_torch_batch(batch)
        result['event_ids'] = batch['event_ids']

        for name in ('region_valid', 'region_absence_code', 'rbp_available'):
            result[name] = torch.as_tensor(batch[name])

        return result

class EventTokenEncoder(nn.Module):

    # Static·RBP-region query의 학습 parameter를 구성하는 함수
    def __init__(self, rbp_names, d_model=32):
        super().__init__()

        if d_model < 1 or not rbp_names or len(set(rbp_names)) != len(rbp_names):
            raise ValueError('Positive dimension and unique RBP names required')

        self.rbp_names = tuple(rbp_names)
        self.d_model = d_model
        self.signature = {
            'static_names': list(STATIC_NAMES),
            'rbp_names': list(rbp_names),
            'region_names': list(REGIONS),
            'rbp_fields': list(RBP_FIELDS),
        }

        self.static_weight = nn.Parameter(torch.empty(len(STATIC_NAMES), d_model))
        self.static_bias = nn.Parameter(torch.empty(len(STATIC_NAMES), d_model))
        self.static_missing = nn.Parameter(torch.empty(len(STATIC_NAMES), d_model))

        for p in (self.static_weight, self.static_bias, self.static_missing):
            nn.init.normal_(p, std=0.02)

        self.rbp_identity = nn.Embedding(len(rbp_names), d_model)
        self.region_identity = nn.Embedding(3, d_model)
        self.binding_projection = nn.Sequential(
            nn.Linear(2 * len(RBP_FIELDS), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.static_norm = nn.LayerNorm(d_model)
        self.binding_norm = nn.LayerNorm(d_model)

        self.register_buffer('rbp_indices', torch.arange(len(rbp_names)).repeat_interleave(3))
        self.register_buffer('region_indices', torch.arange(3).repeat(len(rbp_names)))

    # Checkpoint에 저장할 feature 구성과 차원을 반환하는 함수
    def get_extra_state(self):
        return {'signature': self.signature, 'd_model': self.d_model}

    # Checkpoint의 feature 구성과 차원을 확인하는 함수
    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Checkpoint token vocabulary/order or dimension mismatch')

    # Event feature를 query token과 유효 mask로 변환하는 함수
    def forward(self, batch):
        if batch['signature'] != self.signature:
            raise ValueError('Encoder token vocabulary/order mismatch')

        s, sm = (batch['static_values'], batch['static_observed'])
        r, rm, valid = (batch['rbp_values'], batch['rbp_observed'], batch['query_valid'])
        B, F, T = (s.shape[0], len(STATIC_NAMES), len(self.rbp_names) * 3)

        if (
            (s.shape != (B, F))
            or (sm.shape != s.shape)
            or (r.shape != (B, T, len(RBP_FIELDS)))
            or (rm.shape != r.shape)
            or (valid.shape != (B, F + T))
        ):
            raise ValueError('Incorrect batch dimensions')
        if any((x.dtype != torch.bool for x in (sm, rm, valid))):
            raise ValueError('Masks must be boolean, True means usable')
        if not torch.isfinite(s[sm]).all() or not torch.isfinite(r[rm]).all():
            raise ValueError('Observed input must be finite')

        s = torch.where(sm, s, torch.zeros_like(s))
        r = torch.where(rm, r, torch.zeros_like(r))
        static = s[..., None] * self.static_weight + self.static_bias + (~sm)[..., None] * self.static_missing
        binding = self.binding_projection(torch.cat([r, rm.to(r.dtype)], dim=-1))

        binding = binding + self.rbp_identity(self.rbp_indices) + self.region_identity(self.region_indices)
        tokens = torch.cat([self.static_norm(static), self.binding_norm(binding)], dim=1)
        tokens = tokens.masked_fill(~valid[..., None], 0)

        return {'tokens': tokens, 'query_valid': valid}
