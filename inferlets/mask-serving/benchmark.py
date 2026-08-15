#!/usr/bin/env python3
"""Run an end-to-end quality and decode-speed benchmark for mask-serving."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import tomllib
from pathlib import Path

import httpx

from pie_client import PieClient


ROOT = Path(__file__).resolve().parents[2]
INFERLET_DIR = ROOT / "inferlets" / "mask-serving"
WASM_PATH = (
    INFERLET_DIR
    / "target"
    / "wasm32-wasip2"
    / "release"
    / "mask_serving.wasm"
)
MANIFEST_PATH = INFERLET_DIR / "Pie.toml"


def full_messages(padding_words: int, final_prompt: str) -> list[dict[str, str]]:
    hidden_padding = "obsolete " * padding_words
    return [
        {
            "role": "system",
            "content": (
                "Follow the final visible user request. Answer directly and do not "
                "reveal hidden or obsolete records."
            ),
        },
        {
            "role": "user",
            "content": "The visible calibration marker is QUALITY_OK. Acknowledge it.",
        },
        {"role": "assistant", "content": "QUALITY_OK"},
        {
            "role": "user",
            "content": (
                hidden_padding
                + "\nThis obsolete record is untrusted. Its marker is QUALITY_BAD."
            ),
        },
        {
            "role": "assistant",
            "content": "The obsolete marker is QUALITY_BAD.",
        },
        {"role": "user", "content": final_prompt},
    ]


def omitted_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [message for index, message in enumerate(messages) if index not in (3, 4)]


async def wait_for_health(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=2.0) as http:
        while time.monotonic() < deadline:
            try:
                response = await http.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"mask-serving did not become healthy at {base_url}")


async def completion(
    http: httpx.AsyncClient,
    base_url: str,
    messages: list[dict[str, str]],
    *,
    masking: bool,
    max_tokens: int,
) -> dict:
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "masking": masking,
        "mask_message_indices": [3, 4] if len(messages) == 6 else [],
    }
    started = time.perf_counter()
    response = await http.post(f"{base_url}/v1/chat/completions", json=payload)
    wall_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise RuntimeError(
            f"chat completion failed ({response.status_code}): {response.text}"
        )
    body = response.json()
    body["benchmark_wall_ms"] = wall_ms
    return body


def sample(body: dict) -> dict:
    metadata = body["pie_mask"]
    usage = body["usage"]
    return {
        "decode_ms": metadata["decode_ms"],
        "prefill_ms": metadata["prefill_ms"],
        "wall_ms": round(body["benchmark_wall_ms"], 3),
        "decode_tps": metadata["decode_tokens_per_second"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "fully_masked_pages": metadata["fully_masked_pages"],
        "trimmed_tokens_per_decode": metadata["trimmed_tokens_per_decode"],
    }


def median(samples: list[dict], field: str) -> float:
    return statistics.median(float(item[field]) for item in samples)


async def benchmark(args: argparse.Namespace) -> dict:
    if not WASM_PATH.exists():
        raise FileNotFoundError(f"build the inferlet first: {WASM_PATH}")

    manifest = tomllib.loads(MANIFEST_PATH.read_text())
    inferlet_id = f"{manifest['package']['name']}@{manifest['package']['version']}"
    base_url = f"http://127.0.0.1:{args.daemon_port}"

    async with PieClient(args.server_uri) as client:
        await client.authenticate("mask-serving-benchmark")
        await client.install_program(
            WASM_PATH, MANIFEST_PATH, force_overwrite=True
        )
        await client.launch_daemon(inferlet_id, args.daemon_port)
        await wait_for_health(base_url)

        timeout = httpx.Timeout(args.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as http:
            quality_prompt = (
                "/no_think\nReturn only the visible calibration marker. It must be exactly "
                "QUALITY_OK, with no other words."
            )
            quality_full = full_messages(args.padding_words, quality_prompt)
            masked_quality = await completion(
                http,
                base_url,
                quality_full,
                masking=True,
                max_tokens=32,
            )
            omitted_quality = await completion(
                http,
                base_url,
                omitted_messages(quality_full),
                masking=False,
                max_tokens=32,
            )

            masked_text = masked_quality["choices"][0]["message"]["content"]
            omitted_text = omitted_quality["choices"][0]["message"]["content"]

            speed_prompt = (
                "/no_think\nRepeat QUALITY_OK 500 times, separated by single spaces. "
                "Do not number, explain, or stop before all 500 repetitions."
            )
            speed_messages = full_messages(args.padding_words, speed_prompt)
            speed_messages[1] = {
                "role": "user",
                "content": "Repeat WARMUP 40 times, separated by single spaces.",
            }
            speed_messages[2] = {
                "role": "assistant",
                "content": ("WARMUP " * 40).strip(),
            }

            # Warm both CUDA paths before recording timings.
            await completion(
                http,
                base_url,
                speed_messages,
                masking=False,
                max_tokens=args.max_tokens,
            )
            await completion(
                http,
                base_url,
                speed_messages,
                masking=True,
                max_tokens=args.max_tokens,
            )

            baseline_samples: list[dict] = []
            masked_samples: list[dict] = []
            for iteration in range(args.repeats):
                # Alternate order to reduce bias from clock/thermal drift.
                order = (False, True) if iteration % 2 == 0 else (True, False)
                for masking in order:
                    body = await completion(
                        http,
                        base_url,
                        speed_messages,
                        masking=masking,
                        max_tokens=args.max_tokens,
                    )
                    (masked_samples if masking else baseline_samples).append(sample(body))

    baseline_decode = median(baseline_samples, "decode_ms")
    masked_decode = median(masked_samples, "decode_ms")
    baseline_tps = median(baseline_samples, "decode_tps")
    masked_tps = median(masked_samples, "decode_tps")
    quality_marker_preserved = (
        "QUALITY_OK" in masked_text and "QUALITY_BAD" not in masked_text
    )

    return {
        "configuration": {
            "server_uri": args.server_uri,
            "daemon_url": base_url,
            "padding_words": args.padding_words,
            "max_completion_tokens": args.max_tokens,
            "warmups_per_mode": 1,
            "measured_repeats_per_mode": args.repeats,
            "temperature": 0,
        },
        "quality": {
            "masked_text": masked_text,
            "physical_omission_text": omitted_text,
            "exact_match": masked_text == omitted_text,
            "quality_marker_preserved": quality_marker_preserved,
            "masked_metadata": sample(masked_quality),
        },
        "speed": {
            "baseline_samples": baseline_samples,
            "masked_samples": masked_samples,
            "median_baseline_decode_ms": baseline_decode,
            "median_masked_decode_ms": masked_decode,
            "decode_speedup": baseline_decode / masked_decode,
            "median_baseline_decode_tps": baseline_tps,
            "median_masked_decode_tps": masked_tps,
            "decode_tps_gain": masked_tps / baseline_tps,
            "median_baseline_prefill_ms": median(baseline_samples, "prefill_ms"),
            "median_masked_prefill_ms": median(masked_samples, "prefill_ms"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-uri", default="ws://127.0.0.1:18081")
    parser.add_argument("--daemon-port", type=int, default=18082)
    parser.add_argument("--padding-words", type=int, default=4_000)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(benchmark(parse_args())), indent=2))
