import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
text = args.input.read_text()
data = [json.loads(line) for line in text.splitlines() if line.strip()] if args.input.suffix == '.jsonl' else json.loads(text)
runs = data if isinstance(data, list) else data.get('runs', [data])
ids = set()
for run in runs:
    for key in ('run_id', 'scenario_id', 'variant'):
        if not isinstance(run.get(key), str) or not run[key].strip():
            raise ValueError('Missing ' + key)
    if run['run_id'] in ids:
        raise ValueError('Duplicate run_id')
    ids.add(run['run_id'])
    if not isinstance(run.get('turns'), list):
        raise ValueError('turns must be an array')
    for key in ('latency_ms', 'total_tokens', 'cost_usd'):
        value = run.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < float('inf')):
            raise ValueError('Invalid ' + key)
args.output.write_text(json.dumps({'schema_version':'1.0.0','runs':runs}, indent=2, allow_nan=False))
print(f'Exported {len(runs)} observed runs')
