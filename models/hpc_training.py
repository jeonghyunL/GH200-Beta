import csv
import json
import shutil
import socket
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .multimodal_data import PairDataset, MultimodalCollator, move_batch
from .cross_attention import SplicingCrossAttention
DEVICE = torch.device('cuda:0')

# JSON을 임시 파일에 쓴 뒤 대상 파일로 교체하는 함수
def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

# 모델 학습·검증·테스트를 수행하고 checkpoint와 지표를 저장하는 함수
def run_training(store, output, epochs=1, batch_size=4, learning_rate=0.001, device=DEVICE, num_workers=0):
    device = torch.device(device)

    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU가 할당된 compute node에서 실행하세요.')
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError('epochs/batch_size는 양수, num_workers는 0 이상이어야 합니다.')

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    np.random.seed(42)

    datasets = {split: PairDataset(store, split) for split in ['train', 'validation', 'test']}
    loaders = {split: DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == 'train',
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=False,
        collate_fn=MultimodalCollator(store),
    ) for split, dataset in datasets.items()}
    config = json.loads((store.root / 'data_processed/training_setup/model_config.json').read_text())
    model = SplicingCrossAttention(store.rbp_names, **config['model_kwargs']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    report = {
        'status': 'running',
        'host': socket.gethostname(),
        'GPU': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'CUDA_runtime': torch.version.cuda,
        'epochs_requested': epochs,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'learning_rate': learning_rate,
        'seed': 42,
        'precision': 'float32',
        'optimizer': 'AdamW',
        'weight_decay': 0.01,
        'split_counts': {},
        'history': [],
        'optimizer_steps': 0,
        'scope': 'all prepared valid pairs; no pair/token/window truncation',
    }

    for split, dataset in datasets.items():
        report['split_counts'][split] = {
            'pairs': len(dataset),
            'events': int(len(np.unique(dataset.flat // dataset.n_samples))),
            'samples': int(len(np.unique(dataset.flat % dataset.n_samples))),
        }

    write_json(output / 'run_summary.json', report)
    shutil.copytree(store.root / 'data_processed/training_setup', output / 'training_setup')

    for relative in [
        'data_processed/pilot_training_data_flow.json',
        'reports/reference_genome_provenance.json',
        'bundle_manifest.json',
    ]:
        source = store.root / relative

        if source.exists():
            shutil.copy2(source, output / source.name)

    start = time.perf_counter()
    best = float('inf')

    # 한 split을 처리하고 오차·시간·GPU 메모리를 집계하는 함수
    def run_pass(split, epoch):
        training = split == 'train'
        model.train(training)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        tick = time.perf_counter()
        squared = 0.0
        absolute = 0.0
        count = 0

        for step, cpu_batch in enumerate(loaders[split], 1):
            batch = move_batch(cpu_batch, str(device))

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                prediction = model(batch)['prediction']
                error = prediction - batch['label']
                loss = error.square().mean()

                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if training:
                    loss.backward()
                    optimizer.step()
                    report['optimizer_steps'] += 1

            squared += error.detach().square().sum().item()
            absolute += error.detach().abs().sum().item()
            count += len(prediction)

            if step % 1000 == 0 or step == len(loaders[split]):
                elapsed = time.perf_counter() - tick
                remaining = (len(datasets[split]) - count) * elapsed / max(count, 1) / 60
                print(
                    f'{split} epoch {epoch}: {count:,}/{len(datasets[split]):,} pairs | {elapsed / 60:.1f} min | estimated remaining {remaining:.1f} min',
                    flush=True,
                )
                write_json(
                    output / 'progress.json',
                    {
                        'split': split,
                        'epoch': epoch,
                        'completed_pairs_in_pass': count,
                        'elapsed_seconds': elapsed,
                        'optimizer_steps': report['optimizer_steps'],
                    },
                )

        torch.cuda.synchronize(device)
        assert count == len(datasets[split])
        pass_seconds = time.perf_counter() - tick

        return {
            'split': split,
            'epoch': epoch,
            'pairs_processed': count,
            'mse': squared / count,
            'mae': absolute / count,
            'seconds': pass_seconds,
            'pairs_per_second': count / pass_seconds,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device),
        }

    try:
        for epoch in range(1, epochs + 1):
            report['history'].append(run_pass('train', epoch))
            validation = run_pass('validation', epoch)
            report['history'].append(validation)

            if validation['mse'] < best:
                best = validation['mse']
                torch.save(
                    {'model': model.state_dict(), 'config': model.config, 'epoch': epoch},
                    output / 'best.pt',
                )
                report['best_epoch'] = epoch

            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model.config,
                    'epoch': epoch,
                },
                output / 'last.pt',
            )
            write_json(output / 'run_summary.json', report)

        try:
            checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(output / 'best.pt', map_location=device)

        model.load_state_dict(checkpoint['model'])
        report['history'].append(run_pass('test', report['best_epoch']))
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'interrupted_or_failed'
        report['error'] = str(exc)
        raise
    finally:
        torch.cuda.synchronize(device)
        report['total_seconds'] = time.perf_counter() - start
        write_json(output / 'run_summary.json', report)

    return (model.eval(), report)

# 일부 학습 batch의 처리량과 예상 시간을 측정하는 함수
def benchmark_train_batches(store, batch_size=4, num_workers=0, measured_steps=1000, warmup_steps=50, device=DEVICE):
    device = torch.device(device)
    dataset = PairDataset(store, 'train')
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=MultimodalCollator(store),
    )

    config = json.loads((store.root / 'data_processed/training_setup/model_config.json').read_text())
    model = SplicingCrossAttention(store.rbp_names, **config['model_kwargs']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    iterator = iter(loader)

    measured_pairs = 0
    started = None

    for step in range(warmup_steps + measured_steps):
        cpu_batch = next(iterator)
        batch = move_batch(cpu_batch, str(device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)['prediction']

        loss = (prediction - batch['label']).square().mean()
        loss.backward()
        optimizer.step()

        if step + 1 == warmup_steps:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        elif step + 1 > warmup_steps:
            measured_pairs += len(prediction)

    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    pairs_per_second = measured_pairs / seconds
    estimated_train_minutes = len(dataset) / pairs_per_second / 60

    estimated_total_minutes = estimated_train_minutes * 1.12058422140339
    result = {
        'GPU': torch.cuda.get_device_name(device),
        'batch_size': batch_size,
        'num_workers': num_workers,
        'measured_steps': measured_steps,
        'measured_pairs': measured_pairs,
        'seconds': seconds,
        'train_pairs_per_second': pairs_per_second,
        'estimated_train_minutes': estimated_train_minutes,
        'estimated_total_minutes_scaled_from_A100': estimated_total_minutes,
        'A100_colab_train_pairs_per_second': 324.658971273155,
        'speed_ratio_vs_A100_colab': pairs_per_second / 324.658971273155,
    }
    del model, optimizer, loader, iterator, batch, cpu_batch
    torch.cuda.empty_cache()

    return result

# 한 event–sample pair의 attention과 token 정보를 저장하는 함수
def export_attention(store, model, pair, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cpu = MultimodalCollator(store)([pair])
    model.eval()

    with torch.no_grad():
        result = model(move_batch(cpu, next(model.parameters()).device), return_attention=True)

    q = result['query_valid'][0].cpu().numpy()
    k = result['context_valid'][0].cpu().numpy()
    attention = result['attention'][0].cpu().numpy()[:, q][:, :, k]
    queries = np.array([x['token_name'] for x in store.events.token_manifest()])[q]

    labels = np.array(store.context_labels(cpu), dtype=str)[k]
    kinds = np.array(
        ['SNP'] * cpu['snp']['valid'].shape[1] + ['CpG'] * cpu['cpg']['valid'].shape[1] + ['RBP_expression'] * len(store.rbp_names) + ['fallback'],
    )[k]
    regions = np.array(['upstream', 'target', 'downstream'])
    region_labels = np.concatenate(
        [
            regions[cpu['snp']['regions'][0].numpy()],
            regions[cpu['cpg']['regions'][0].numpy()],
            np.array([''] * (len(store.rbp_names) + 1)),
        ],
    )[k]

    np.savez_compressed(
        output / 'attention.npz',
        attention=attention,
        query_names=queries,
        context_names=labels,
        context_kind=kinds,
        context_region=region_labels,
    )

    with (output / 'top_attention.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query', 'context', 'kind', 'region', 'mean_head_attention'])

        for query, scores in zip(queries, attention.mean(axis=0)):
            for j in np.argsort(-scores)[:10]:
                writer.writerow([query, labels[j], kinds[j], region_labels[j], float(scores[j])])

    write_json(
        output / 'example.json',
        {
            'event_id': cpu['event_ids'][0],
            'donor_id': cpu['donor_ids'][0],
            'event_index': int(pair[0]),
            'sample_index': int(pair[1]),
            'PSI': float(cpu['label'][0]),
            'predicted_PSI': float(result['prediction'][0]),
            'attention_shape': list(attention.shape),
            'note': 'Attention is not a correlation or causal effect.',
        },
    )

    return (attention, queries, labels)
