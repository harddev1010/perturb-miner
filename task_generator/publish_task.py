from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        elif char == "#" and quote is None:
            return value[:index].strip()
    return value.strip()


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / "task_generator.env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = _unquote(_strip_inline_comment(value))


def main() -> None:
    load_local_env()
    from task_generator.generator import generate_and_publish_task

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Generate and publish one Perturb task")
    parser.add_argument("--state-path", default=os.getenv("PERTURB_TASK_GENERATOR_STATE", "task_generator_state.json"))
    parser.add_argument("--status", default="open", choices=("open", "disabled", "validating"))
    args = parser.parse_args()
    task = generate_and_publish_task(state_path=args.state_path, status=args.status)
    print(f"task_id={task.task_id}")
    print(f"image_url={task.image_url}")
    print(f"image_id={task.image_id}")
    print(f"true_label={task.true_label}")


if __name__ == "__main__":
    main()
