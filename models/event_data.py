import csv
import hashlib
import json
from pathlib import Path
import numpy as np
EVENT_TYPES = ('SE', 'RI', 'MX', 'A5', 'A3', 'AF', 'AL')
REGIONS = ('upstream', 'target', 'downstream')
STATIC_NAMES = (
    'MaxEnt_5',
    'MaxEnt_3',
    'MaxEnt_5_Mask',
    'MaxEnt_3_Mask',
    'ESE_Count',
    'ESE_Density',
    'ESS_Count',
    'ESS_Density',
    'Upstream_ISE_Count',
    'Upstream_ISE_Density',
    'Downstream_ISE_Count',
    'Downstream_ISE_Density',
    'Upstream_ISS_Count',
    'Upstream_ISS_Density',
    'Downstream_ISS_Count',
    'Downstream_ISS_Density',
    'Exon_Len',
    'Upstream_I_Len',
    'Downstream_I_Len',
    'Log_Exon_Len',
    'Log_Upstream_I_Len',
    'Log_Downstream_I_Len',
    'Exon_GC',
    'Upstream_I_GC',
    'Downstream_I_GC',
    'Upstream_Proximal_GC',
    'Downstream_Proximal_GC',
    'PhyloP_Exon',
    'PhyloP_Up_Intron',
    'PhyloP_Down_Intron',
    'PhyloP_Up_Proximal',
    'PhyloP_Down_Proximal',
    *EVENT_TYPES,
)
RBP_FIELDS = (
    'has_peak',
    'peak_count',
    'distance_to_start_bp',
    'distance_to_end_bp',
    'region_length_bp',
    'peak_density_per_kb',
)

STATIC_BINARY = np.array([name.endswith('_Mask') or name in EVENT_TYPES for name in STATIC_NAMES])
STATIC_LOG = np.array(
    [name.endswith('_Count') or name in ('Exon_Len', 'Upstream_I_Len', 'Downstream_I_Len') for name in STATIC_NAMES],
)

class EventFeatureStore:

    # Event feature를 읽고 ID·shape·관측값을 검증하는 함수
    def __init__(self, static_csv, rbp_npz):
        with np.load(rbp_npz, allow_pickle=False) as f:
            self.rbp = {name: f[name] for name in f.files}

        a = self.rbp
        self.event_ids = tuple(a['event_ids'].tolist())
        self.rbp_names = tuple(a['rbp_names'].tolist())
        self.static_names = STATIC_NAMES

        self.event_index = {name: i for i, name in enumerate(self.event_ids)}
        E, R = (len(self.event_ids), len(self.rbp_names))

        if not E or not R or len(self.event_index) != E or (len(set(self.rbp_names)) != R):
            raise ValueError('Empty or duplicate event/RBP identities')
        if tuple(a['region_names']) != REGIONS:
            raise ValueError('Unexpected region order')

        shapes = {name: (E, R, 3) for name in RBP_FIELDS if name != 'region_length_bp'}
        shapes.update(
            region_length_bp=(E, 3),
            region_valid=(E, 3),
            region_absence_code=(E, 3),
            rbp_available=(R,),
            event_types=(E,),
            strands=(E,),
        )

        for name, shape in shapes.items():
            if a[name].shape != shape:
                raise ValueError(f'{name}: expected shape {shape}')

        for name in ('has_peak', 'region_valid', 'rbp_available'):
            if a[name].dtype != np.bool_:
                raise ValueError(f'{name} must be boolean')

        if not np.array_equal(a['has_peak'], a['peak_count'] > 0):
            raise ValueError('Peak presence/count mismatch')

        valid = a['region_valid'][:, None, :] & a['rbp_available'][None, :, None]

        if np.any(a['has_peak'] & ~valid):
            raise ValueError('Peak in excluded/unavailable slot')
        if np.any(a['region_valid'] & (a['region_length_bp'] <= 0)):
            raise ValueError('Valid region requires positive length')
        if not np.array_equal(a['region_absence_code'] == 0, a['region_valid']):
            raise ValueError('Region validity/absence code mismatch')

        for name in ('distance_to_start_bp', 'distance_to_end_bp'):
            values = a[name]

            if (
                (not np.isfinite(values[a['has_peak']]).all())
                or (np.any(values[a['has_peak']] < 0))
                or (not np.isnan(values[~a['has_peak']]).all())
            ):
                raise ValueError(f'{name}: invalid distance/missingness')

        density = a['peak_density_per_kb']

        if (
            (not np.isfinite(density[valid]).all())
            or (np.any(density[valid] < 0))
            or (not np.isnan(density[~valid]).all())
        ):
            raise ValueError('Invalid density/missingness')

        self.static = np.empty((E, len(STATIC_NAMES)), dtype=np.float32)
        seen = set()

        with Path(static_csv).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)

            if (
                (len(reader.fieldnames or []) != 40)
                or (set(reader.fieldnames) != {'event_id', *STATIC_NAMES})
            ):
                raise ValueError('Expected event_id and the 39 known static features')

            for row in reader:
                eid = row['event_id']

                if eid in seen or eid not in self.event_index:
                    raise ValueError(f'Duplicate or extra static event: {eid}')

                seen.add(eid)
                self.static[self.event_index[eid]] = [float(row[n]) if row[n] else np.nan for n in STATIC_NAMES]

        if seen != set(self.event_ids):
            raise ValueError('Missing static events; partial joins are forbidden')
        if np.isinf(self.static).any():
            raise ValueError('Infinite static feature')
        if not np.isin(self.static[:, STATIC_BINARY], [0, 1]).all():
            raise ValueError('Event indicators and MaxEnt masks must be 0/1')

        types = self.static[:, -len(EVENT_TYPES):]

        if (
            (not np.all(types.sum(axis=1) == 1))
            or (not np.array_equal(np.array(EVENT_TYPES)[types.argmax(axis=1)], a['event_types']))
        ):
            raise ValueError('Static and RBP event types disagree')

        self.static_observed = np.isfinite(self.static)

        for score, mask in [('MaxEnt_5', 'MaxEnt_5_Mask'), ('MaxEnt_3', 'MaxEnt_3_Mask')]:
            self.static_observed[:, STATIC_NAMES.index(score)] &= self.static[:, STATIC_NAMES.index(mask)] == 1

        self.signature = {
            'static_names': list(STATIC_NAMES),
            'rbp_names': list(self.rbp_names),
            'region_names': list(REGIONS),
            'rbp_fields': list(RBP_FIELDS),
        }

    # Event ID를 배열 index로 변환하는 함수
    def indices(self, event_ids):
        return np.array([self.event_index[e] for e in event_ids], dtype=np.int64)

    # 선택한 event의 feature와 mask를 묶는 함수
    def batch(self, indices):
        idx = np.asarray(indices)

        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or (not len(idx)):
            raise ValueError('Use a nonempty 1D integer index list')
        if np.any(idx < 0) or np.any(idx >= len(self.event_ids)):
            raise IndexError('Event index out of bounds')

        a, B, R = (self.rbp, len(idx), len(self.rbp_names))
        region_valid = a['region_valid'][idx]
        valid = np.broadcast_to(region_valid[:, None, :] & a['rbp_available'][None, :, None], (B, R, 3))
        values = np.stack(
            [
                a['has_peak'][idx],
                a['peak_count'][idx],
                a['distance_to_start_bp'][idx],
                a['distance_to_end_bp'][idx],
                np.broadcast_to(a['region_length_bp'][idx, None, :], (B, R, 3)),
                a['peak_density_per_kb'][idx],
            ],
            axis=-1,
        ).astype(
            np.float32,
        )

        observed = np.broadcast_to(valid[..., None], values.shape).copy()
        observed[..., 2:4] &= a['has_peak'][idx, ..., None]
        observed &= np.isfinite(values)

        return {
            'event_ids': tuple((self.event_ids[i] for i in idx)),
            'signature': self.signature,
            'static_values': self.static[idx].copy(),
            'static_observed': self.static_observed[idx].copy(),
            'rbp_values': values.reshape(B, R * 3, len(RBP_FIELDS)),
            'rbp_observed': observed.reshape(B, R * 3, len(RBP_FIELDS)),
            'query_valid': np.concatenate(
                [np.ones((B, len(STATIC_NAMES)), dtype=bool), valid.reshape(B, R * 3)],
                axis=1,
            ),
            'region_valid': region_valid.copy(),
            'region_absence_code': a['region_absence_code'][idx].copy(),
            'rbp_available': a['rbp_available'].copy(),
        }

    # Event token의 이름과 속성을 반환하는 함수
    def token_manifest(self):
        rows = [{
            'token_index': i,
            'token_name': name,
            'kind': 'legacy_static',
            'rbp': '',
            'region': '',
            'geometry': 'legacy_39d',
            'values': name,
        } for i, name in enumerate(STATIC_NAMES)]

        for rbp in self.rbp_names:
            for region in REGIONS:
                rows.append(
                    {
                        'token_index': len(rows),
                        'token_name': f'{rbp}_{region}',
                        'kind': 'rbp_binding',
                        'rbp': rbp,
                        'region': region,
                        'geometry': 'user_diagram_v1',
                        'values': ','.join(RBP_FIELDS),
                    },
                )

        return rows

# Static·binding 값에 고정 변환을 적용하는 함수
def transformed_values(batch):
    s = np.where(batch['static_observed'], batch['static_values'], 0).astype(np.float64)
    r = np.where(batch['rbp_observed'], batch['rbp_values'], 0).astype(np.float64)

    if np.any(s[:, STATIC_LOG] < 0) or np.any(r < 0):
        raise ValueError('Counts, lengths and binding features must be nonnegative')

    s[:, STATIC_LOG] = np.log1p(s[:, STATIC_LOG])
    r[..., 1:] = np.log1p(r[..., 1:])

    return (s, r)

class EventNormalizer:

    # 학습 event에서 정규화 통계를 계산하는 함수
    def fit(self, store, training_event_ids, batch_size=512):
        ids = list(training_event_ids)

        if not ids or len(set(ids)) != len(ids) or batch_size < 1:
            raise ValueError('Provide nonempty unique training event IDs and positive batch size')

        idx = store.indices(ids)
        accum = [[np.zeros(n), np.zeros(n), np.zeros(n)] for n in (len(STATIC_NAMES), len(RBP_FIELDS))]

        for start in range(0, len(idx), batch_size):
            b = store.batch(idx[start:start + batch_size])

            for values, mask, state in zip(transformed_values(b), (b['static_observed'], b['rbp_observed']), accum):
                values, mask = (values.reshape(-1, values.shape[-1]), mask.reshape(-1, mask.shape[-1]))
                n = mask.sum(axis=0)
                mean = np.divide(
                    (values * mask).sum(axis=0),
                    n,
                    out=np.zeros_like(n, dtype=float),
                    where=n > 0,
                )
                m2 = ((values - mean) ** 2 * mask).sum(axis=0)

                total = state[0] + n
                delta = mean - state[1]
                ratio = np.divide(n, total, out=np.zeros_like(mean), where=total > 0)
                state[2] += m2 + delta ** 2 * state[0] * ratio

                state[1] += delta * ratio
                state[0] = total

        self.stats = {}

        for name, (n, mean, m2), binary in zip(
            ('static', 'rbp'),
            accum,
            (STATIC_BINARY, np.array([True, False, False, False, False, False])),
        ):
            scale = np.sqrt(np.divide(m2, n, out=np.zeros_like(m2), where=n > 0))
            scale[scale < 1e-08] = 1
            mean[binary], scale[binary] = (0, 1)
            self.stats[name] = {'mean': mean, 'scale': scale, 'observations': n.astype(np.int64)}

        self.signature = store.signature
        self.training_event_count = len(ids)
        self.training_event_ids_sha256 = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()

        return self

    # 저장된 통계로 event feature를 정규화하는 함수
    def transform(self, batch):
        if not hasattr(self, 'stats'):
            raise RuntimeError('Fit on explicitly chosen training events or load saved statistics first')
        if batch['signature'] != self.signature:
            raise ValueError('Normalizer vocabulary/order mismatch')

        result = dict(batch)

        for name, values in zip(('static', 'rbp'), transformed_values(batch)):
            stats = self.stats[name]
            scaled = (values - stats['mean']) / stats['scale']
            result[name + '_values'] = np.where(batch[name + '_observed'], scaled, 0).astype(np.float32)

            if not np.isfinite(result[name + '_values']).all():
                raise ValueError('Nonfinite normalized input')

        result['normalized'] = True

        return result

    # 정규화 통계와 feature 정보를 저장하는 함수
    def save(self, path):
        if not hasattr(self, 'stats'):
            raise RuntimeError('Cannot save unfitted normalizer')

        payload = {
            'version': 1,
            'signature': self.signature,
            'training_event_count': self.training_event_count,
            'training_event_ids_sha256': self.training_event_ids_sha256,
            'stats': {k: {n: v.tolist() for n, v in s.items()} for k, s in self.stats.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # 저장된 정규화 통계를 읽고 검증하는 함수
    @classmethod

    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))

        if payload['version'] != 1:
            raise ValueError('Unsupported normalizer version')

        obj = cls()
        obj.signature = payload['signature']
        obj.training_event_count = payload['training_event_count']
        obj.training_event_ids_sha256 = payload['training_event_ids_sha256']

        obj.stats = {k: {n: np.asarray(v) for n, v in s.items()} for k, s in payload['stats'].items()}

        for name, size in [('static', len(STATIC_NAMES)), ('rbp', len(RBP_FIELDS))]:
            stats = obj.stats[name]

            if (
                (any((v.shape != (size,) or not np.isfinite(v).all() for v in stats.values())))
                or (np.any(stats['scale'] <= 0))
            ):
                raise ValueError('Invalid saved normalizer statistics')

        return obj
