"""Sequential PIE benchmark; run after the server finishes loading.

Dependencies: pip install transformers websockets msgpack blake3
Build examples/text-completion-bench for wasm32-wasip3 first.
"""
import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'packages/python/client/src'))
from pie_client import Event, PieClient


async def request(client, prompt, count, *, tokens=None, ignore_eos=True):
    start = time.perf_counter()
    first = None
    process = await client.launch_process('text-completion-bench@0.1.0', dict(
        prompt=prompt, prompt_tokens=tokens, max_tokens=count, temperature=0,
        ignore_eos=ignore_eos, return_text=True, report_timing=True, no_think=True,
    ))
    try:
        while True:
            event, value = await asyncio.wait_for(process.recv(), timeout=900)
            if event == Event.Message and value == 't0' and first is None:
                first = time.perf_counter()
            if event == Event.Error:
                raise RuntimeError(value)
            if event != Event.Return:
                continue
            end = time.perf_counter()
            data = json.loads(value) if isinstance(value, (str, bytes)) else value
            n = data['num_output_tokens']
            gaps = data.get('intertoken_us', [])
            if first is None:
                raise RuntimeError('Missing first-token event')
            if ignore_eos and (n != count or len(gaps) != n - 1):
                raise RuntimeError(f'Expected {count} outputs and {count-1} gaps, got {n}, {len(gaps)}')
            return dict(
                prompt_tokens=data['num_prompt_tokens'], output_tokens=n,
                ttft_ms=1000*(first-start), latency_s=end-start,
                output_tokens_s=n/(end-start),
                decode_tokens_s=(n-1)/(sum(gaps)/1e6) if gaps and sum(gaps) else None,
                guest_ttft_us=data.get('ttft_us'), intertoken_us=gaps,
                token_ids=data.get('token_ids'), text=data.get('text', ''),
            )
    finally:
        client.process_event_queues.pop(process.process_id, None)


async def main(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    async with PieClient(args.url) as client:
        await client.install_program(args.wasm, version='0.1.0', force_overwrite=True)
        smoke = await request(client, 'What is 17 multiplied by 23? Give only the numeric answer.', 32, ignore_eos=False)
        (args.output / f'{args.label}-smoke.json').write_text(json.dumps(smoke, indent=2))
        print('SMOKE', smoke['text'], flush=True)
        if smoke['text'].strip() != '391':
            raise RuntimeError('Arithmetic smoke failed; inspect the saved output before benchmarking')
        for length in (128, 1024):
            def ids(index):
                text = f'Benchmark sample {index}. ' + 'The quick brown fox jumps over the lazy dog. ' * (length//8+30)
                return tok.encode(text, add_special_tokens=False)[:length]
            await request(client, '', 256, tokens=ids(10000))
            batch = []
            for index in range(args.requests):
                row = await request(client, '', 256, tokens=ids(index))
                assert row['prompt_tokens'] == length
                batch.append(row)
                print(f'{args.label} prompt={length} request={index} TTFT={row["ttft_ms"]:.2f}ms decode={row["decode_tokens_s"]:.2f}tok/s', flush=True)
            summary = dict(
                prompt_tokens=length, output_tokens=256, concurrency=1,
                requests=len(batch),
                ttft_median_ms=statistics.median(r['ttft_ms'] for r in batch),
                decode_tokens_s_median=statistics.median(r['decode_tokens_s'] for r in batch),
                aggregate_output_tokens_s=sum(r['output_tokens'] for r in batch)/sum(r['latency_s'] for r in batch),
                raw=batch,
            )
            results.append(summary)
            (args.output / f'{args.label}-results.json').write_text(json.dumps(dict(
                runtime='PIE cuda_native', label=args.label,
                protocol='Same raw token IDs, 256 outputs, greedy, ignore EOS; fresh KV per request; 256-output warmup excluded; client TTFT and guest intertoken drain gaps',
                results=results,
            ), indent=2))
            print(json.dumps({k:v for k,v in summary.items() if k != 'raw'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--wasm', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='ws://127.0.0.1:8082')
    p.add_argument('--requests', type=int, default=10)
    asyncio.run(main(p.parse_args()))
