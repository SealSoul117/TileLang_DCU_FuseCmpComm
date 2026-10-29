"""Aggregate matched four-rank event samples without averaging away slow PEs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.directory.glob('round*/[arg][rsg][0-9]*')):
        paths = [directory / f'rank{r}.json' for r in range(4)]
        if not all(p.is_file() for p in paths):
            raise RuntimeError(f'incomplete rank records: {directory}')
        ranks = [json.loads(p.read_text()) for p in paths]
        first = ranks[0]
        assert [r['rank'] for r in ranks] == list(range(4))
        for record in ranks:
            for field in ('shape', 'kind', 'configs', 'repeats', 'warmup', 'job', 'compute_blocks', 'comm_blocks'):
                assert record[field] == first[field], (directory, field)
            assert all(not s['foreign'] for s in record['gpu_process_snapshots'])
        timings = {}
        for variant in ('A00', 'B10', 'C11'):
            samples = [r['samples_ms'][variant] for r in ranks]
            assert all(len(s) == first['repeats'] for s in samples)
            maxima = [max(t) for t in zip(*samples)]
            ordered = sorted(maxima)
            timings[variant] = {'median_ms': statistics.median(maxima),
                'p10_ms': ordered[int(.1*(len(ordered)-1))],
                'p90_ms': ordered[int(.9*(len(ordered)-1))],
                'max_rank_samples_ms': maxima}
        rows.append({'run': str(directory.relative_to(args.directory)),
            'kind': first['kind'], 'shape': first['shape'],
            'compute_blocks': first['compute_blocks'], 'comm_blocks': first['comm_blocks'],
            'timings': timings})
    assert rows, 'no completed rounds found'
    (args.directory / 'summary.json').write_text(json.dumps(rows, indent=2))
    print('| run | M/N/K | compute/comm | A00 ms | B10 ms | C11 ms | A→B speedup | B→C speedup | A→C speedup |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        a,b,c = (row['timings'][v]['median_ms'] for v in ('A00','B10','C11'))
        print(f"| {row['run']} | {'/'.join(map(str,row['shape']))} | {row['compute_blocks']}/{row['comm_blocks']} | {a:.6f} | {b:.6f} | {c:.6f} | {a/b:.3f}× | {b/c:.3f}× | {a/c:.3f}× |")
    pooled = []
    print('\nTwo rounds pooled (60 max-rank event samples per variant):\n')
    print('| kind | M/N/K | compute/comm | A00 ms | B10 ms | C11 ms | A→B speedup | B→C speedup | A→C speedup |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for kind in ('rs', 'ar', 'ag'):
        for size in (2048, 4096, 8192, 16384):
            matched = [r for r in rows if r['kind'] == kind and r['shape'][0] == size]
            assert len(matched) == 2, (kind, size, 'expected two rounds')
            row = matched[0]
            medians = {v: statistics.median([s for r in matched for s in r['timings'][v]['max_rank_samples_ms']])
                       for v in ('A00','B10','C11')}
            a,b,c = (medians[v] for v in ('A00','B10','C11'))
            pooled.append({'kind': kind, 'shape': row['shape'], 'median_ms': medians,
                'hoist_speedup': a/b, 'vector_increment_speedup': b/c, 'combined_speedup': a/c})
            print(f"| {kind} | {'/'.join(map(str,row['shape']))} | {row['compute_blocks']}/{row['comm_blocks']} | {a:.6f} | {b:.6f} | {c:.6f} | {a/b:.3f}× | {b/c:.3f}× | {a/c:.3f}× |")
    (args.directory / 'pooled_summary.json').write_text(json.dumps(pooled, indent=2))


if __name__ == '__main__':
    main()
