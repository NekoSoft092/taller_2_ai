from __future__ import annotations

import random
from collections.abc import Callable

from algorithms.base import EvaluationFunction
from algorithms.base.search import finish_search_root, is_cutoff, legal_actions
from engine.model import Team, TeamAction
from engine.rules import step
from engine.state import GameState


def expectimax_search(
    state: GameState,
    depth: int,
    evaluation_function: EvaluationFunction,
    rng: random.Random,
    *,
    prob: float = 0.0,
    on_expand: Callable[[], None] | None = None,
) -> tuple[TeamAction, TeamAction, float]:
    """
    Depth-limited expectimax from the root: Colombia MAX, mixed rival, and expected value.

    The rival is a chance node: with probability prob it acts uniformly at random;
    otherwise it plays the greedy MIN reply. The root returns Colombia's action and the
    greedy rival reply used to break ties at the root.

    Tips:
    - Same ply shell as minimax_search; the rival layer becomes an expectation.
    - For each Colombia action, score every rival reply with ply(successor, depth - 1)[2].
    - expected = (1 - prob) * min(scores) + prob * mean(scores).
    - Colombia still picks max over expected values.
    """

    def ply(state: GameState, depth: int) -> tuple[TeamAction | None, TeamAction | None, float]:
        if on_expand is not None:
            on_expand()

        if is_cutoff(state, depth):
            return None, None, evaluation_function(state)

        colombia_actions = legal_actions(state, Team.COLOMBIA)
        rival_actions = legal_actions(state, Team.RIVAL)

        ### YOUR CODE HERE ###
        # --- SOLUTION START ---
        best_colombia_action: TeamAction | None = None
        best_rival_action: TeamAction | None = None
        best_value = float("-inf")

        for colombia_action in colombia_actions:
            rival_scores: list[tuple[TeamAction, float]] = []
            for rival_action in rival_actions:
                successor = step(state, colombia_action, rival_action)
                value = ply(successor, depth - 1)[2]
                rival_scores.append((rival_action, value))

            greedy_rival_action, min_value = min(rival_scores, key=lambda item: item[1])
            mean_value = sum(value for _action, value in rival_scores) / len(rival_scores)
            expected_value = (1 - prob) * min_value + prob * mean_value

            if expected_value > best_value:
                best_colombia_action = colombia_action
                best_rival_action = greedy_rival_action
                best_value = expected_value

        return best_colombia_action, best_rival_action, best_value
        # --- SOLUTION END ---

    return finish_search_root(*ply(state, depth))
