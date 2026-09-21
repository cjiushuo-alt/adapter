"""Create the fixed-format LIBERO-4 aggregate table from suite summaries."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "results" / "hpcm3_libero4"
SUITES = (
    ("Spatial", "spatial"),
    ("Object", "object"),
    ("Goal", "goal"),
    ("LIBERO-10", "libero_10"),
)


def main() -> None:
    rows = []
    details = []
    for display_name, directory in SUITES:
        path = RESULT_ROOT / directory / "summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            f"| {display_name:<9} | {data['success_episodes']:>7} | "
            f"{data['total_episodes']:>8} | {data['exact_train_frame_matches']:>25} |"
        )
        details.append(
            f"- {display_name}: success rate {data['success_rate']:.4%}; "
            f"action unnorm key `{data['action_unnorm_key']}`."
        )

    output = RESULT_ROOT / "summary.md"
    content = "\n".join(
        [
            "# HPCM3 LIBERO-4 Evaluation",
            "",
            "| Suite     | Success | Episodes | Exact Train-Frame Matches |",
            "| --------- | ------: | -------: | ------------------------: |",
            *rows,
            "",
            *details,
            "",
        ]
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(output)


if __name__ == "__main__":
    main()
