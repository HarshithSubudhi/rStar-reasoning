"""
rStar Demo: End-to-End Reasoning with MCTS & Mutual Discrimination
Paper: "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" (ICLR 2025)
"""

import os
import sys
import json
import argparse
from rstar import rStarAgent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run rStar reasoning on benchmark problems."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "toy_dataset.json"),
        help="Path to benchmark JSON dataset."
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=12,
        help="Number of MCTS rollouts per problem (default: 12)."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum depth of reasoning tree (default: 4)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 80)
    print(" rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers")
    print(" ICLR 2025 Paper Implementation")
    print("=" * 80)

    if not os.path.exists(args.data):
        print(f"[Error] Dataset file not found at: {args.data}")
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"\nLoaded {len(problems)} problem(s) from: {args.data}")
    print(f"MCTS Configuration: rollouts={args.rollouts}, max_depth={args.depth}\n")

    agent = rStarAgent(max_rollouts=args.rollouts, max_depth=args.depth)
    total_passed = 0

    for idx, item in enumerate(problems, 1):
        q_id = item.get("id", f"problem_{idx}")
        question = item["question"]
        expected = item.get("expected_answer", "")
        domain = item.get("domain", "General Reasoning")

        print("-" * 80)
        print(f"[{idx}/{len(problems)}] ID: {q_id} | Domain: {domain}")
        print(f"Problem: {question}")
        if expected:
            print(f"Expected Answer: {expected}")
        print("-" * 80)

        # Run MCTS + Mutual Discrimination pipeline
        result = agent.solve(question)

        selected_answer = result["selected_answer"]
        confidence = result["confidence"]
        disc_score = result.get("discriminator_score", 0.0)
        freq_score = result.get("frequency_score", 0.0)
        candidates_count = result["candidates_count"]
        best_trajectory = result["best_trajectory"]

        is_correct = str(selected_answer).strip() == str(expected).strip() if expected else None
        if is_correct:
            total_passed += 1

        print("\n>>> Selected Consensus Trajectory (MCTS + 5 Cognitive Actions):")
        print(best_trajectory)
        print("\n>>> Mutual Evaluation Metrics:")
        print(f"  * Generated Trajectories          : {candidates_count}")
        print(f"  * Discriminator Verification Score: {disc_score:.4f}")
        print(f"  * Ensemble Consistency Vote       : {freq_score:.4f}")
        print(f"  * Mutual Confidence Score S(T)    : {confidence:.4f}")
        print(f"  * Model Prediction                : '{selected_answer}'")
        if expected:
            status_tag = "[PASS] CORRECT" if is_correct else "[FAIL] MISMATCH"
            print(f"  * Validation Status               : {status_tag}")
        print()

    print("=" * 80)
    if problems and "expected_answer" in problems[0]:
        acc = (total_passed / len(problems)) * 100
        print(f" SUMMARY: {total_passed}/{len(problems)} Solved Correctly ({acc:.1f}% Accuracy)")
    else:
        print(f" Completed reasoning rollouts on {len(problems)} benchmark items.")
    print("=" * 80)

if __name__ == "__main__":
    main()
