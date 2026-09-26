"""Run Neural Painter autoresearch with autonomous code mutation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from painter.autocode import autonomous_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/autocode"))
    parser.add_argument("--mutation-cycles", type=int, default=3)
    parser.add_argument("--experiment-iterations", type=int, default=6)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--palette-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--model-base-url",
        default=os.environ.get("NEURAL_PAINTER_MODEL_BASE_URL", "http://localhost:11434/v1"),
        help="OpenAI-compatible /v1 endpoint, e.g. Ollama/LM Studio/vLLM",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEURAL_PAINTER_MODEL", "qwen2.5-coder:7b"),
        help="model identifier served by the configured endpoint",
    )
    parser.add_argument(
        "--model-api-key",
        default=os.environ.get("NEURAL_PAINTER_MODEL_API_KEY", ""),
    )
    parser.add_argument("--model-timeout", type=int, default=180)
    parser.add_argument("--agent-turns", type=int, default=24)
    parser.add_argument("--gate-timeout", type=int, default=900)
    parser.add_argument("--experiment-timeout", type=int, default=3600)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="fast-forward the current branch to accepted autonomous mutations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    summary = autonomous_research(
        repo_root,
        args.image,
        args.output_root.resolve(),
        mutation_cycles=args.mutation_cycles,
        experiment_iterations=args.experiment_iterations,
        budget=args.budget,
        palette_size=args.palette_size,
        seed=args.seed,
        device=args.device,
        model_base_url=args.model_base_url,
        model_name=args.model,
        model_api_key=args.model_api_key,
        model_timeout=args.model_timeout,
        agent_turns=args.agent_turns,
        gate_timeout=args.gate_timeout,
        experiment_timeout=args.experiment_timeout,
        apply=args.apply,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
