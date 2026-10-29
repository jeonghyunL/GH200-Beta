import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .event_data import EventFeatureStore, EventNormalizer
from .event_encoder import to_torch_batch

# NPZ에서 지정한 배열을 읽는 함수
def load_arrays(path, names=None):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in (f.files if names is None else names)}

# Donor를 학습·검증·테스트 집합으로 나누는 함수
def donor_split(donor_ids, seed=42):
    ids = np.asarray(donor_ids)

    if len(set(ids)) != len(ids) or len(ids) < 10:
        raise ValueError('Expected at least ten unique donors, one sample per donor')

    order = np.random.default_rng(seed).permutation(np.argsort(ids))
    n = len(ids)
    a = int(0.8 * n)
    b = a + int(0.1 * n)

    return {name: order[lo:hi].tolist() for name, lo, hi in [('train', 0, a), ('validation', a, b), ('test', b, n)]}

class ExpressionNormalizer:

    # 학습 donor의 발현량으로 RBP별 정규화 통계를 계산하는 함수
    def fit(self, values, observed, train_indices, donor_ids, rbp_names):
        ix = np.asarray(train_indices, dtype=np.int64)

        if not len(ix) or len(set(ix.tolist())) != len(ix):
            raise ValueError('Unique training samples required')
        if ix.min() < 0 or ix.max() >= len(values):
            raise ValueError('Sample index out of bounds')

        v = np.log1p(np.where(observed[ix], values[ix], 0)).astype(np.float64)
        m = observed[ix]

        if not np.isfinite(v[m]).all():
            raise ValueError('Invalid abundance')

        count = m.sum(axis=0)
        self.mean = np.divide((v * m).sum(axis=0), count, out=np.zeros(v.shape[1]), where=count > 0)
        variance = np.divide(
            ((v - self.mean) ** 2 * m).sum(axis=0),
            count,
            out=np.zeros(v.shape[1]),
            where=count > 0,
        )
        self.scale = np.sqrt(variance)

        self.scale[self.scale < 1e-08] = 1
        self.rbp_names = list(rbp_names)
        self.training_donors = sorted(np.asarray(donor_ids)[ix].tolist())
        self.observations = count

        return self

    # 발현량에 로그 변환과 RBP별 표준화를 적용하는 함수
    def transform(self, values, observed, rbp_names):
        if list(rbp_names) != self.rbp_names:
            raise ValueError('RBP vocabulary mismatch')

        result = np.where(observed, (np.log1p(np.where(observed, values, 0)) - self.mean) / self.scale, 0).astype(
            np.float32,
        )

        if not np.isfinite(result).all():
            raise ValueError('Nonfinite normalized expression')

        return result

    # 발현 정규화 통계와 donor 정보를 저장하는 함수
    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    'version': 1,
                    'mean': self.mean.tolist(),
                    'scale': self.scale.tolist(),
                    'rbp_names': self.rbp_names,
                    'training_donors': self.training_donors,
                    'observations': self.observations.tolist(),
                    'transform': 'log1p then per-RBP training-donor z-score',
                },
                indent=2,
            ),
            encoding='utf-8',
        )

    # 저장된 발현 정규화 통계를 읽고 검증하는 함수
    @classmethod

    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))

        if data['version'] != 1:
            raise ValueError('Unsupported expression statistics version')

        obj = cls()
        obj.mean = np.asarray(data['mean'])
        obj.scale = np.asarray(data['scale'])
        obj.rbp_names = data['rbp_names']

        obj.training_donors = data['training_donors']
        obj.observations = np.asarray(data['observations'])

        if (
            (len(obj.mean) != len(obj.rbp_names))
            or (not np.isfinite(obj.mean).all())
            or (not np.isfinite(obj.scale).all())
            or (np.any(obj.scale <= 0))
        ):
            raise ValueError('Invalid expression statistics')

        return obj

class MultimodalStore:

    # PSI와 omics 배열을 읽고 ID·좌표·배열 정렬을 검증하는 함수
    def __init__(self, root):
        self.root = Path(root)
        p = self.root / 'data_processed'
        self.index = load_arrays(p / 'pilot_wgs_candidate_index.npz')
        self.event_ids = self.index['event_ids']

        self.donor_ids = self.index['donor_ids']
        self.events = EventFeatureStore(self.root / 'chr19_static_39d_final_v5.csv', p / 'event_rbp_features.npz')
        self.event_indices = self.index['event_indices']
        self.rbp_names = self.events.rbp_names

        if not np.array_equal(np.asarray(self.events.event_ids)[self.event_indices], self.event_ids):
            raise ValueError('Event order mismatch')

        with np.load(p / 'event_psi_exact_draft.npz', allow_pickle=False) as f:
            if not np.array_equal(
                f['sample_ids'][self.index['rna_sample_indices']],
                self.index['rna_sample_ids'],
            ):
                raise ValueError('RNA order mismatch')

            self.psi = f['psi'][np.ix_(self.event_indices, self.index['rna_sample_indices'])].astype(
                np.float32,
            )

        self.pair_valid = np.isfinite(self.psi)

        if not np.array_equal(self.pair_valid, self.index['pair_valid']):
            raise ValueError('PSI mask mismatch')
        if np.any((self.psi[self.pair_valid] < 0) | (self.psi[self.pair_valid] > 1)):
            raise ValueError('PSI outside [0,1]')

        self.snp = load_arrays(
            p / 'snp_context_grch38.npz',
            [
                'variant_ids',
                'position0',
                'ref',
                'alt',
                'dosage',
                'gq',
                'dp',
                'donor_ids',
                'rna_sample_ids',
                'wgs_sample_ids',
                'genome_build',
            ],
        )
        self.snp_links = load_arrays(
            p / 'event_snp_context_links.npz',
            [
                'event_ids',
                'event_indptr',
                'variant_indices',
                'region_codes',
                'distance_from_target_start',
                'distance_from_target_end',
                'variant_ids',
                'genome_build',
            ],
        )
        self.cpg = load_arrays(
            p / 'cpg_context_grch38.npz',
            [
                'probe_ids',
                'position0',
                'beta',
                'observed',
                'donor_ids',
                'rna_sample_ids',
                'methylation_sample_ids',
                'genome_build',
            ],
        )
        self.cpg_links = load_arrays(
            p / 'event_cpg_context_links.npz',
            [
                'event_ids',
                'event_indptr',
                'probe_indices',
                'region_codes',
                'distance_from_target_start',
                'distance_from_target_end',
                'probe_ids',
                'genome_build',
            ],
        )

        self.cpg_samples = self.index['joint_sample_indices']

        if not np.array_equal(self.snp['donor_ids'], self.donor_ids):
            raise ValueError('SNP donor order')
        if not np.array_equal(self.snp['wgs_sample_ids'], self.index['wgs_sample_ids']):
            raise ValueError('SNP sample order')
        if not np.array_equal(self.cpg['donor_ids'][self.cpg_samples], self.donor_ids):
            raise ValueError('CpG donor alignment')

        for key in ['rna_sample_ids', 'methylation_sample_ids']:
            if not np.array_equal(self.cpg[key][self.cpg_samples], self.index[key]):
                raise ValueError('CpG sample alignment')

        if not np.array_equal(self.snp['rna_sample_ids'], self.index['rna_sample_ids']):
            raise ValueError('SNP/RNA alignment')

        for data, links, key in [(self.snp, self.snp_links, 'variant_ids'), (self.cpg, self.cpg_links, 'probe_ids')]:
            if data['genome_build'].item() != 'GRCh38' or links['genome_build'].item() != 'GRCh38':
                raise ValueError('Context build mismatch')
            if (
                (not np.array_equal(data[key], links[key]))
                or (not np.array_equal(links['event_ids'], self.event_ids))
            ):
                raise ValueError('Context identity/order mismatch')
            if (
                (len(links['event_indptr']) != len(self.event_ids) + 1)
                or (links['event_indptr'][0] != 0)
                or (np.any(np.diff(links['event_indptr']) < 0))
            ):
                raise ValueError('Invalid ragged offsets')

            indices = links['variant_indices' if key == 'variant_ids' else 'probe_indices']

            if (
                (links['event_indptr'][-1] != len(indices))
                or (np.any(indices < 0))
                or (np.any(indices >= len(data[key])))
            ):
                raise ValueError('Invalid site references')

        exp = load_arrays(
            p / 'rbp_expression_context.npz',
            ['sample_ids', 'rbp_names', 'abundance', 'observed'],
        )

        if (
            (tuple(exp['rbp_names']) != self.rbp_names)
            or (not np.array_equal(
                exp['sample_ids'][self.index['rna_sample_indices']],
                self.index['rna_sample_ids'],
            ))
        ):
            raise ValueError('Expression identity/order')

        self.expression = exp['abundance'][self.index['rna_sample_indices']]
        self.expression_observed = exp['observed'][self.index['rna_sample_indices']]
        self.snp_alleles = np.stack(
            [[{'A': 0, 'C': 1, 'G': 2, 'T': 3}[a] for a in self.snp[k]] for k in ['ref', 'alt']],
            axis=-1,
        )

    # Event에 연결된 SNP 또는 CpG의 index·영역·거리를 반환하는 함수
    def links_for(self, kind, event):
        links = self.snp_links if kind == 'snp' else self.cpg_links
        a, b = links['event_indptr'][event:event + 2]
        key = 'variant_indices' if kind == 'snp' else 'probe_indices'

        return (
            links[key][a:b],
            links['region_codes'][a:b],
            links['distance_from_target_start'][a:b],
            links['distance_from_target_end'][a:b],
        )

    # Donor split과 normalizer를 읽고 적용하는 함수
    def load_preparation(self, directory=None):
        directory = Path(directory) if directory else self.root / 'data_processed/training_setup'
        split = json.loads((directory / 'donor_split.json').read_text(encoding='utf-8'))

        if (
            (split['donor_ids'] != self.donor_ids.tolist())
            or (split['event_ids_sha256'] != hashlib.sha256('\n'.join(self.event_ids).encode()).hexdigest())
        ):
            raise ValueError('Split identities do not match store')

        all_indices = [i for name in ['train', 'validation', 'test'] for i in split['sample_indices'][name]]

        if sorted(all_indices) != list(range(len(self.donor_ids))):
            raise ValueError('Split overlaps or omits donors')

        self.split = split
        self.event_normalizer = EventNormalizer.load(directory / 'event_normalizer.json')
        self.expression_normalizer = ExpressionNormalizer.load(directory / 'expression_normalizer.json')
        train = split['sample_indices']['train']

        if self.expression_normalizer.training_donors != sorted(self.donor_ids[train].tolist()):
            raise ValueError('Expression fit donor mismatch')

        train_events = self.event_ids[self.pair_valid[:, train].any(axis=1)]
        digest = hashlib.sha256('\n'.join(sorted(train_events)).encode()).hexdigest()

        if self.event_normalizer.training_event_ids_sha256 != digest:
            raise ValueError('Event fit set mismatch')

        self.normalized_expression = self.expression_normalizer.transform(
            self.expression,
            self.expression_observed,
            self.rbp_names,
        )

        return self

    # Batch의 context token에 대응하는 이름을 반환하는 함수
    def context_labels(self, batch, row=0):
        result = []

        for p in batch['snp']['indices'][row].tolist():
            result.append(None if p < 0 else str(self.snp['variant_ids'][p]))

        for p in batch['cpg']['indices'][row].tolist():
            result.append(None if p < 0 else str(self.cpg['probe_ids'][p]))

        return result + [f'{rbp}_expression' for rbp in self.rbp_names] + ['empty_context_fallback']

class PairDataset(Dataset):

    # 선택한 split에서 유효 PSI pair의 index를 구성하는 함수
    def __init__(self, store, split):
        if split not in ['train', 'validation', 'test']:
            raise ValueError('Unknown split')

        allowed = np.zeros(len(store.donor_ids), dtype=bool)
        allowed[store.split['sample_indices'][split]] = True
        self.flat = np.flatnonzero(store.pair_valid & allowed[None, :])
        self.n_samples = len(store.donor_ids)

    # 유효 event–sample pair 수를 반환하는 함수
    def __len__(self):
        return len(self.flat)

    # 지정한 pair의 event·sample index를 반환하는 함수
    def __getitem__(self, i):
        e, s = divmod(int(self.flat[i]), self.n_samples)

        return (e, s)

# 거리의 부호를 유지하며 로그 변환하는 함수
def signed_log_bp(values):
    v = np.asarray(values, dtype=np.float32)

    return np.sign(v) * np.log10(1 + np.abs(v))

class MultimodalCollator:

    # Batch 구성에 사용할 multimodal store를 보관하는 함수
    def __init__(self, store):
        self.store = store

    # Event–sample pair를 padding과 mask가 포함된 batch로 만드는 함수
    def __call__(self, pairs):
        if not pairs:
            raise ValueError('Empty batch')

        st = self.store
        ee = np.asarray([p[0] for p in pairs])
        ss = np.asarray([p[1] for p in pairs])
        B = len(pairs)

        if (
            (np.any(ee < 0))
            or (np.any(ee >= len(st.event_ids)))
            or (np.any(ss < 0))
            or (np.any(ss >= len(st.donor_ids)))
        ):
            raise IndexError('Pair out of range')
        if not st.pair_valid[ee, ss].all():
            raise ValueError('Attempt to load missing PSI label')

        event = to_torch_batch(st.event_normalizer.transform(st.events.batch(st.event_indices[ee])))
        batch = {
            'event': event,
            'label': torch.from_numpy(st.psi[ee, ss]),
            'event_indices': torch.from_numpy(ee),
            'sample_indices': torch.from_numpy(ss),
            'event_ids': st.event_ids[ee].tolist(),
            'donor_ids': st.donor_ids[ss].tolist(),
            'rbp_names': list(st.rbp_names),
        }

        for kind, width in [('snp', 9), ('cpg', 5)]:
            lists = [st.links_for(kind, int(e)) for e in ee]
            L = max((len(x[0]) for x in lists))
            values = np.zeros((B, L, width), dtype=np.float32)
            valid = np.zeros((B, L), dtype=bool)

            indices = np.full((B, L), -1, dtype=np.int64)
            regions = np.zeros((B, L), dtype=np.int64)
            alleles = np.zeros((B, L, 2), dtype=np.int64)

            for b, (ids, codes, ds, de) in enumerate(lists):
                n = len(ids)
                sample = int(ss[b])
                indices[b, :n] = ids
                regions[b, :n] = codes

                if kind == 'snp':
                    dose = st.snp['dosage'][ids, sample]
                    gq = st.snp['gq'][ids, sample]
                    dp = st.snp['dp'][ids, sample]
                    obs = dose >= 0

                    gm = gq >= 0
                    dm = dp >= 0
                    valid[b, :n] = obs
                    values[b, :n] = np.stack(
                        [
                            np.where(obs, dose, 0) / 2,
                            np.log1p(np.maximum(gq, 0)) / np.log(100),
                            np.log1p(np.maximum(dp, 0)) / np.log(100),
                            st.snp['position0'][ids] / 58617616,
                            signed_log_bp(ds),
                            signed_log_bp(de),
                            obs,
                            gm,
                            dm,
                        ],
                        axis=-1,
                    )

                    alleles[b, :n] = st.snp_alleles[ids]
                else:
                    source_sample = st.cpg_samples[sample]
                    obs = st.cpg['observed'][ids, source_sample]
                    beta = st.cpg['beta'][ids, source_sample]
                    valid[b, :n] = obs

                    values[b, :n] = np.stack(
                        [
                            np.where(obs, beta, 0),
                            st.cpg['position0'][ids] / 58617616,
                            signed_log_bp(ds),
                            signed_log_bp(de),
                            obs,
                        ],
                        axis=-1,
                    )

            batch[kind] = {
                'values': torch.from_numpy(values),
                'valid': torch.from_numpy(valid),
                'indices': torch.from_numpy(indices),
                'regions': torch.from_numpy(regions),
            }

            if kind == 'snp':
                batch[kind]['alleles'] = torch.from_numpy(alleles)

        obs = st.expression_observed[ss]
        batch['expression'] = {
            'values': torch.from_numpy(
                np.stack([st.normalized_expression[ss], obs], axis=-1).astype(np.float32),
            ),
            'valid': torch.from_numpy(obs.copy()),
        }

        return batch

# Batch 안의 tensor를 지정한 device로 이동하는 함수
def move_batch(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}

    return batch
