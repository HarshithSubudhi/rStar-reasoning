"""
Definition of the 5 Human-like Reasoning Actions in rStar
Reference: Section 3.2 of "Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers" (ICLR 2025)
"""

from enum import Enum
from typing import List, Dict, Any

class ActionType(str, Enum):
    PROPOSE_ONE_STEP = "Propose a one-step thought"
    COMPLETE_REASONING = "Complete reasoning thought"
    PROPOSE_SUBQUESTION = "Propose subquestions and answer"
    REANSWER_SUBQUESTION = "Re-answer the subquestion"
    REPHRASE_QUESTION = "Rephrase the question"

ACTION_PROMPT_TEMPLATES: Dict[ActionType, str] = {
    ActionType.PROPOSE_ONE_STEP: (
        "Given the original problem and the partial reasoning steps so far:\n"
        "Problem: {question}\n"
        "Partial Steps:\n{history}\n"
        "Task: Propose only the next immediate logical step to make progress. Do not finish everything yet."
    ),
    ActionType.COMPLETE_REASONING: (
        "Given the original problem and the partial steps:\n"
        "Problem: {question}\n"
        "Partial Steps:\n{history}\n"
        "Task: Directly complete all remaining steps and provide the final concise answer formatted as 'Final Answer: <val>'."
    ),
    ActionType.PROPOSE_SUBQUESTION: (
        "Given the problem and the current progress:\n"
        "Problem: {question}\n"
        "Partial Steps:\n{history}\n"
        "Task: Break down the next barrier into an explicit sub-question (e.g., 'Subquestion: ...') and solve it ('Subanswer: ...')."
    ),
    ActionType.REANSWER_SUBQUESTION: (
        "Given the problem and the subquestion previously tackled:\n"
        "Problem: {question}\n"
        "Partial Steps:\n{history}\n"
        "Task: Re-examine the previous sub-question and propose an alternative calculation or check for any hidden assumptions."
    ),
    ActionType.REPHRASE_QUESTION: (
        "Original Problem: {question}\n"
        "Task: Rephrase the question statement clearly, identifying the exact variables given and what needs to be solved."
    )
}

def get_action_list() -> List[ActionType]:
    """Returns the 5 canonical reasoning actions in rStar."""
    return [
        ActionType.PROPOSE_ONE_STEP,
        ActionType.COMPLETE_REASONING,
        ActionType.PROPOSE_SUBQUESTION,
        ActionType.REANSWER_SUBQUESTION,
        ActionType.REPHRASE_QUESTION
    ]
