"""
rStar Agent: High-level orchestrator for Mutual Reasoning
Reference: "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" (ICLR 2025)
"""

from typing import Dict, Any, Optional
from .generator import MCTSGenerator
from .discriminator import MutualDiscriminator

class rStarAgent:
    """
    End-to-End rStar Reasoning Agent.
    Orchestrates the Generator (MCTS with 5 cognitive actions) and Discriminator (Peer Verification).
    """
    def __init__(
        self,
        generator: Optional[MCTSGenerator] = None,
        discriminator: Optional[MutualDiscriminator] = None,
        max_rollouts: int = 10,
        max_depth: int = 4
    ):
        self.generator = generator or MCTSGenerator(max_rollouts=max_rollouts, max_depth=max_depth)
        self.discriminator = discriminator or MutualDiscriminator()

    def solve(self, question: str) -> Dict[str, Any]:
        """
        Executes the full rStar pipeline on a given question:
        1. Explore reasoning trajectories using MCTS + 5 cognitive actions.
        2. Evaluate trajectories using the Mutual Discriminator.
        3. Select top consensus trajectory.
        """
        # Step 1: MCTS Rollouts (Generation)
        candidate_terminals = self.generator.search(question)

        # Step 2: Peer Discrimination & Mutual Consistency Scoring
        ranked_trajectories = self.discriminator.evaluate_trajectories(question, candidate_terminals)

        if not ranked_trajectories:
            return {
                "question": question,
                "selected_answer": None,
                "confidence": 0.0,
                "best_trajectory": "No valid trajectory found.",
                "all_candidates": []
            }

        top_choice = ranked_trajectories[0]

        return {
            "question": question,
            "selected_answer": top_choice["final_answer"],
            "confidence": top_choice["mutual_score"],
            "discriminator_score": top_choice["discriminator_score"],
            "frequency_score": top_choice["frequency_score"],
            "best_trajectory": top_choice["full_trajectory"],
            "candidates_count": len(ranked_trajectories),
            "all_candidates": ranked_trajectories
        }
