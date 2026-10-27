"""
MCTS Tree Node Data Structure for rStar Framework
Reference: "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" (ICLR 2025)
"""

import math
from typing import List, Optional, Dict, Any

class MCTSNode:
    """
    Represents a single reasoning state in the Monte Carlo Tree Search.
    Each node represents an intermediate reasoning thought or sub-question resolution.
    """
    def __init__(
        self,
        thought: str,
        action_type: str,
        parent: Optional['MCTSNode'] = None,
        depth: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.thought: str = thought
        self.action_type: str = action_type  # One of the 5 rStar reasoning actions
        self.parent: Optional['MCTSNode'] = parent
        self.children: List['MCTSNode'] = []
        self.depth: int = depth
        self.metadata: Dict[str, Any] = metadata or {}

        # MCTS Statistics
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.is_terminal: bool = False
        self.final_answer: Optional[str] = None
        self.discriminator_score: float = 0.0

    @property
    def value(self) -> float:
        """Returns the average estimated value Q(s)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """
        Computes Upper Confidence Bound for Trees (UCT):
        UCT(s) = Q(s) + c * sqrt(ln(N(parent)) / N(s))
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        if self.parent is None or self.parent.visits == 0:
            return self.value
        
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def add_child(self, child_node: 'MCTSNode') -> 'MCTSNode':
        """Appends a child node to this state."""
        child_node.parent = self
        child_node.depth = self.depth + 1
        self.children.append(child_node)
        return child_node

    def get_trajectory(self) -> List['MCTSNode']:
        """Returns the full sequence of reasoning nodes from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_full_reasoning_text(self) -> str:
        """Joins reasoning thoughts along the path."""
        trajectory = self.get_trajectory()
        steps = []
        for i, node in enumerate(trajectory):
            if node.thought and node.thought != "ROOT":
                steps.append(f"Step {i} ({node.action_type}): {node.thought}")
        return "\n".join(steps)

    def is_fully_expanded(self, max_children: int = 5) -> bool:
        """Checks if the node has spawned max action branches."""
        return len(self.children) >= max_children
