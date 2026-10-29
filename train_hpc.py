from models.cross_attention import SplicingCrossAttention
from models.hpc_training import run_training, benchmark_train_batches, export_attention

# 입력 로딩부터 학습과 결과 출력까지 실행하는 함수
def main():
    import os
    from pathlib import Path
    RUN_ROOT = Path('/scratch/t617a02/GH200_chr19_compare')
    os.environ['ST_PILOT_ROOT'] = str(RUN_ROOT / 'multimodal_colab_full')
    os.environ['ST_OUTPUT_ROOT'] = str(RUN_ROOT / 'results_hpc')
    print(os.environ['ST_PILOT_ROOT'])

    print(os.environ['ST_OUTPUT_ROOT'])
    from pathlib import Path
    import hashlib
    import json
    import os
    import socket
    import sys
    import time
    import numpy as np
    import torch
    ROOT = Path(os.environ.get('ST_PILOT_ROOT', '/data/jhyun.l/ST_pilot/multimodal_colab_full')).expanduser().resolve()
    OUTPUT_ROOT = Path(os.environ.get('ST_OUTPUT_ROOT', str(ROOT / 'results_hpc'))).expanduser().resolve()
    VERIFY_MANIFEST = False
    assert ROOT.is_dir(), f'프로젝트 루트가 없습니다: {ROOT}'

    assert (ROOT / 'models').is_dir(), f"models 디렉터리가 없습니다: {ROOT / 'models'}"
    assert torch.cuda.is_available(), 'CUDA GPU가 보이지 않습니다. 로그인 노드가 아니라 GPU가 할당된 compute node에서 Jupyter kernel을 실행했는지 확인하세요.'
    DEVICE = torch.device('cuda:0')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print('host:', socket.gethostname())
    print('project root:', ROOT)
    print('output root:', OUTPUT_ROOT)
    print('python:', sys.version.split()[0])

    print('torch:', torch.__version__)
    print('CUDA runtime:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(DEVICE))
    properties = torch.cuda.get_device_properties(DEVICE)

    print('GPU memory (GiB):', round(properties.total_memory / 1024 ** 3, 2))

    if VERIFY_MANIFEST:
        manifest_path = ROOT / 'bundle_manifest.json'
        assert manifest_path.exists(), f'manifest가 없습니다: {manifest_path}'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        for name, expected in manifest.items():
            path = ROOT / name
            assert path.is_file(), f'입력 파일이 없습니다: {path}'
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected, f'checksum 불일치: {name}'

        print('bundle manifest 검증 완료')
    else:
        print('bundle manifest 검증 생략; 필요하면 VERIFY_MANIFEST=True로 변경')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from models.multimodal_data import MultimodalStore, PairDataset
    store = MultimodalStore(ROOT).load_preparation()

    for split in ['train', 'validation', 'test']:
        dataset = PairDataset(store, split)
        print(split, 'samples:', len(store.split['sample_indices'][split]), 'pairs:', len(dataset))

    print('입력 로딩 완료')
    from torch import nn
    from torch.utils.data import DataLoader
    from models.event_data import STATIC_NAMES, REGIONS, RBP_FIELDS
    from models.multimodal_data import MultimodalCollator, move_batch
    import numpy as np
    input_dataset = PairDataset(store, 'train')
    input_batch = MultimodalCollator(store)([input_dataset[i] for i in range(2)])

    for group in ['event', 'snp', 'cpg', 'expression']:
        print(f'\n{group}')

        for name, value in input_batch[group].items():
            if isinstance(value, torch.Tensor):
                print(f'  {name:20s} {str(tuple(value.shape)):18s} {value.dtype}')

    print('label / 실제 PSI:', input_batch['label'])
    print('event ID:', input_batch['event_ids'])
    print('donor ID:', input_batch['donor_ids'])
    config = json.loads((ROOT / 'data_processed/training_setup/model_config.json').read_text())

    # 중첩된 출력에서 tensor shape를 추출하는 함수
    def describe_shape(value):
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, dict):
            return {k: describe_shape(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe_shape(v) for v in value]

        return type(value).__name__

    # 모듈 출력 shape를 표시하는 hook을 만드는 함수
    def shape_hook(name):

        # 모듈 이름과 출력 shape를 표시하는 함수
        def hook(module, inputs, output):
            print(name, '→', describe_shape(output))

        return hook

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        preview_model = SplicingCrossAttention(store.rbp_names, **config['model_kwargs']).eval()
        print(preview_model)
        print('학습 파라미터 수:', sum((p.numel() for p in preview_model.parameters())))

        names = [
            'event_encoder',
            'context_encoder',
            'cross_attention',
            'attention_norm',
            'ff',
            'ff_norm',
            'predictor',
        ]
        handles = [preview_model.get_submodule(name).register_forward_hook(shape_hook(name)) for name in names]

        try:
            with torch.no_grad():
                preview_result = preview_model(input_batch, return_attention=True)

            print('최종 PSI shape:', tuple(preview_result['prediction'].shape))
            print('Attention shape:', tuple(preview_result['attention'].shape))
        finally:
            for handle in handles:
                handle.remove()

    del preview_model, preview_result, input_batch, input_dataset
    RUN_BENCHMARK = False
    BENCHMARK_STEPS = 1000
    BATCH_SIZE = 4

    NUM_WORKERS = 0

    if RUN_BENCHMARK:
        benchmark = benchmark_train_batches(
            store,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            measured_steps=BENCHMARK_STEPS,
        )
        print(json.dumps(benchmark, indent=2, ensure_ascii=False))

    from datetime import datetime
    EPOCHS = 1
    BATCH_SIZE = 4
    NUM_WORKERS = 0

    OUTPUT = OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('결과 폴더:', OUTPUT)
    model, report = run_training(
        store,
        OUTPUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    import pandas as pd
    metrics = pd.DataFrame(report['history'])
    metrics['minutes'] = metrics['seconds'] / 60
    metrics['peak_GPU_GiB'] = metrics['peak_gpu_allocated_bytes'] / 1024 ** 3

    print(metrics[['split', 'epoch', 'pairs_processed', 'mse', 'mae', 'minutes', 'peak_GPU_GiB']])
    print('학습+검증+테스트+저장 총 시간(분):', report['total_seconds'] / 60)
    import numpy as np
    test_samples = store.split['sample_indices']['test']

    site_count = np.diff(store.snp_links['event_indptr']) + np.diff(store.cpg_links['event_indptr'])
    eligible = np.flatnonzero(store.pair_valid[:, test_samples].any(axis=1))
    event_index = int(eligible[np.argmax(site_count[eligible])])
    sample_index = next((s for s in test_samples if store.pair_valid[event_index, s]))

    assert sample_index in test_samples, '해석 예시는 test donor에서 선택하세요.'
    assert store.pair_valid[event_index, sample_index], 'PSI가 없는 조합입니다.'
    attention, query_names, context_names = export_attention(
        store,
        model,
        (event_index, sample_index),
        OUTPUT / f'attention_event{event_index}_sample{sample_index}',
    )
    print(store.event_ids[event_index], store.donor_ids[sample_index])

    print('attention shape [head, query, context]:', attention.shape)
    print(
        pd.read_csv(
            OUTPUT / f'attention_event{event_index}_sample{sample_index}' / 'top_attention.csv',
        ).head(
            30,
        ),
    )
    import matplotlib.pyplot as plt
    mean_attention = attention.mean(axis=0)

    top = np.argsort(-mean_attention.mean(axis=0))[:30]
    plt.figure(figsize=(16, max(8, len(query_names) * 0.11)))
    plt.imshow(mean_attention[:, top], aspect='auto', interpolation='nearest')
    plt.xticks(range(len(top)), context_names[top], rotation=90, fontsize=7)

    plt.yticks(range(len(query_names)), query_names, fontsize=5)
    plt.colorbar(label='Mean attention across heads')
    plt.title('Selected test pair: top 30 contexts (all valid queries)')
    plt.tight_layout()

    plt.savefig(
        OUTPUT / f'attention_event{event_index}_sample{sample_index}' / 'heatmap.png',
        dpi=160,
    )
    plt.show()
    a100 = {
        'train_seconds': 4958.572355746,
        'total_seconds': 5556.497942536001,
        'train_pairs_per_second': 324.658971273155,
        'peak_allocated_bytes': 33120768,
    }
    gh200_train = next((item for item in report['history'] if item['split'] == 'train'))

    gh200_train_pps = gh200_train['pairs_processed'] / gh200_train['seconds']
    comparison = {
        'GH200_train_seconds': gh200_train['seconds'],
        'GH200_total_seconds': report['total_seconds'],
        'GH200_train_pairs_per_second': gh200_train_pps,
        'GH200_training_GPU_hours': report['total_seconds'] / 3600,
        'GH200_peak_allocated_MiB': gh200_train['peak_gpu_allocated_bytes'] / 1024 ** 2,
        'speed_ratio_GH200_over_A100': gh200_train_pps / a100['train_pairs_per_second'],
        'total_time_ratio_GH200_over_A100': report['total_seconds'] / a100['total_seconds'],
        'time_reduction_percent': 100 * (1 - report['total_seconds'] / a100['total_seconds']),
    }
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
