"""Collect Object simulator validation frames outside the official init pool.

Only fresh env.reset() states are rendered. Official .pruned_init states are
read solely to reject state-hash collisions; none is rendered or used to tune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
for root in (REPO, REPO / "LIBERO"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_image, get_libero_wrist_image,
)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--resets-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    rows = []
    seen_states = set()
    for task_id in range(suite.n_tasks):
        protected = {digest(state) for state in suite.get_task_init_states(task_id)}
        env, task = get_libero_env(suite.get_task(task_id), "openvla", resolution=256)
        try:
            env.seed(args.seed + task_id)
            for reset_index in range(args.resets_per_task):
                observation = env.reset()
                state_hash = digest(env.get_sim_state())
                if state_hash in protected or state_hash in seen_states:
                    raise RuntimeError(f"Fresh simulator state collided with protected/previous state: task={task_id}, reset={reset_index}")
                seen_states.add(state_hash)
                for view, rgb in (("agent", get_libero_image(observation)),
                                  ("wrist", get_libero_wrist_image(observation))):
                    rgb = np.ascontiguousarray(rgb)
                    assert rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3)
                    relative = f"task{task_id:02d}_reset{reset_index:02d}_{view}.png"
                    Image.fromarray(rgb).save(args.output_dir / relative)
                    rows.append({"suite": "libero_object", "task_id": task_id,
                                 "task_description": task, "reset_index": reset_index,
                                 "seed": args.seed + task_id, "state_sha256": state_hash,
                                 "view": view, "rgb_sha256": digest(rgb), "path": relative})
        finally:
            env.close()
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "source": "fresh_env_reset_not_default_pruned_init", "seed": args.seed,
        "resets_per_task": args.resets_per_task, "protected_state_collisions": 0,
        "samples": rows,
    }, indent=2) + "\n")
    print(f"Saved {len(rows)} independent simulator-view images in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
