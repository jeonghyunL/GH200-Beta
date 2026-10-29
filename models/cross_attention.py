import torch
from torch import nn
from .event_encoder import EventTokenEncoder

class ContextEncoder(nn.Module):

    # SNP·CpG·발현 token의 projection과 embedding을 구성하는 함수
    def __init__(self, rbp_names, d_model):
        super().__init__()
        self.rbp_names = tuple(rbp_names)
        self.snp = nn.Sequential(nn.Linear(9, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cpg = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.expression = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        self.ref = nn.Embedding(4, d_model)
        self.alt = nn.Embedding(4, d_model)
        self.region = nn.Embedding(3, d_model)

        self.rbp_identity = nn.Embedding(len(rbp_names), d_model)
        self.kind = nn.Embedding(4, d_model)
        self.norm = nn.LayerNorm(d_model)

    # Context token과 유효 mask를 생성하는 함수
    def forward(self, batch):
        if tuple(batch['rbp_names']) != self.rbp_names:
            raise ValueError('Context RBP vocabulary/order mismatch')

        pieces = []
        masks = []

        for kind, name in enumerate(['snp', 'cpg', 'expression']):
            b = batch[name]
            valid = b['valid']
            values = b['values']

            if valid.dtype != torch.bool or values.shape[:2] != valid.shape:
                raise ValueError('Invalid context mask/shape')

            values = torch.where(valid[..., None], values, torch.zeros_like(values))

            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite observed context')

            tokens = getattr(self, name)(values)

            if name in ['snp', 'cpg']:
                regions = torch.where(valid, b['regions'], torch.zeros_like(b['regions']))
                tokens = tokens + self.region(regions)
            if name == 'snp':
                alleles = torch.where(valid[..., None], b['alleles'], torch.zeros_like(b['alleles']))
                tokens = tokens + self.ref(alleles[..., 0]) + self.alt(alleles[..., 1])
            if name == 'expression':
                if valid.shape[1] != len(self.rbp_names):
                    raise ValueError('Expression token count')

                tokens = tokens + self.rbp_identity.weight[None, :, :]

            tokens = self.norm(tokens + self.kind.weight[kind])
            pieces.append(tokens.masked_fill(~valid[..., None], 0))
            masks.append(valid)

        valid = torch.cat(masks, dim=1)
        tokens = torch.cat(pieces, dim=1)
        empty = ~valid.any(dim=1, keepdim=True)
        fallback = self.norm(self.kind.weight[3]).view(1, 1, -1).expand(tokens.shape[0], 1, -1)

        tokens = torch.cat([tokens, fallback.masked_fill(~empty[..., None], 0)], dim=1)

        return (tokens, torch.cat([valid, empty], dim=1))

class SplicingCrossAttention(nn.Module):

    # Event·context encoder와 attention 및 PSI head를 구성하는 함수
    def __init__(self, rbp_names, d_model=32, n_heads=4, dropout=0.1):
        super().__init__()

        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')

        self.config = {'rbp_names': list(rbp_names), 'd_model': d_model, 'n_heads': n_heads, 'dropout': dropout}
        self.event_encoder = EventTokenEncoder(rbp_names, d_model)
        self.context_encoder = ContextEncoder(rbp_names, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.attention_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

        self.predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    # Checkpoint에 저장할 모델 설정을 반환하는 함수
    def get_extra_state(self):
        return self.config

    # Checkpoint의 모델 설정이 일치하는지 확인하는 함수
    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('Checkpoint architecture/vocabulary differs')

    # Event query와 context를 결합해 PSI를 예측하는 함수
    def forward(self, batch, return_attention=False):
        q = self.event_encoder(batch['event'])
        queries = q['tokens']
        qvalid = q['query_valid']
        keys, kvalid = self.context_encoder(batch)

        attended, weights = self.cross_attention(
            queries,
            keys,
            keys,
            key_padding_mask=~kvalid,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        h = self.attention_norm(queries + self.dropout(attended)).masked_fill(~qvalid[..., None], 0)
        h = self.ff_norm(h + self.dropout(self.ff(h))).masked_fill(~qvalid[..., None], 0)
        pooled = h.sum(dim=1) / qvalid.sum(dim=1, keepdim=True).clamp_min(1)

        result = {
            'prediction': torch.sigmoid(self.predictor(pooled).squeeze(-1)),
            'query_valid': qvalid,
            'context_valid': kvalid,
            'query_states': h,
        }

        if return_attention:
            weights = weights.masked_fill(~qvalid[:, None, :, None], 0).masked_fill(
                ~kvalid[:, None, None, :],
                0,
            )
            result['attention'] = weights

        return result
