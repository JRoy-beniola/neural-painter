"""Compare one-shot bounded mutation behavior across local models."""

from __future__ import annotations

import argparse
from pathlib import Path

from painter.agent.model import OpenAICompatibleModel
from painter.code_loop import apply_proposal, propose_function_patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    model = OpenAICompatibleModel(
        args.base_url,
        args.model,
        timeout=args.timeout,
    )
    proposal = propose_function_patch(
        model,
        root,
        path="painter/rich.py",
        function_name="adaptive_primitive_fractions",
        objective=(
            "Improve one existing docstring or comment so the allocator current "
            "behavior is clearer. Do not change executable behavior."
        ),
        behavior_preserving=True,
    )

    print(proposal.patch, end="")
    if args.apply:
        print(apply_proposal(root, proposal))


if __name__ == "__main__":
    main()
