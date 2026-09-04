"""Record native vLLM KV events during a separate cache diagnostic run.

Start before vLLM with the same IPC endpoint/topic in --kv-events-config.
Sequence gaps invalidate complete-eviction claims. No server code is patched.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from pathlib import Path

import msgspec
import zmq
from vllm.distributed.kv_events import KVEventBatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--topic", default="codepin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    decoder = msgspec.msgpack.Decoder(KVEventBatch)
    counts = Counter()
    blocks = Counter()
    first_sequence = last_sequence = None
    gaps = []
    removed = set()
    restored_blocks = 0
    started = time.monotonic()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    socket.setsockopt(zmq.RCVHWM, 100_000)
    socket.connect(args.endpoint)
    (args.output / "ready").write_text(str(time.time()))
    try:
        with (args.output / "events.jsonl").open("w") as stream:
            while not stopped and time.monotonic() - started < args.duration:
                if not socket.poll(250):
                    continue
                topic, sequence_bytes, payload = socket.recv_multipart()
                if topic != args.topic.encode():
                    continue
                sequence = int.from_bytes(sequence_bytes, "big")
                if first_sequence is None:
                    first_sequence = sequence
                elif sequence != last_sequence + 1:
                    gaps.append([last_sequence, sequence])
                last_sequence = sequence
                batch = decoder.decode(payload)
                stream.write(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "received_elapsed_seconds": time.monotonic() - started,
                            "batch": json.loads(msgspec.json.encode(batch)),
                        }
                    )
                    + "\n"
                )
                for event in batch.events:
                    kind = type(event).__name__
                    counts[kind] += 1
                    hashes = getattr(event, "block_hashes", ())
                    group = getattr(event, "group_idx", None)
                    blocks[f"{kind}:group={group}"] += len(hashes)
                    if kind == "BlockRemoved":
                        removed.update((group, value) for value in hashes)
                    elif kind == "BlockStored":
                        for value in hashes:
                            key = (group, value)
                            if key in removed:
                                restored_blocks += 1
                                removed.remove(key)
                    elif kind == "AllBlocksCleared":
                        # Explicit reset is separate from capacity eviction.
                        removed.clear()
    finally:
        socket.close(linger=0)
        context.term()
        summary = {
            "config": {**vars(args), "output": str(args.output)},
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "sequence_gaps": gaps,
            "complete_from_start": first_sequence == 0 and not gaps,
            "event_counts": counts,
            "block_hash_counts": blocks,
            "stored_again_after_removal": restored_blocks,
            "duration_seconds": time.monotonic() - started,
            "interpretation": "BlockRemoved is the native cache removal event. Stored-again counts hashed blocks republished after removal, not recomputed token timing. Explicit AllBlocksCleared is counted separately. Events supplement, not replace, request/cache-token metrics.",
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
