from __future__ import annotations

import random
from collections.abc import Callable

from algorithms.base import EvaluationFunction
from algorithms.base.search import finish_search_root, is_cutoff, legal_actions
from engine.model import Team, TeamAction
from engine.rules import step
from engine.state import GameState


def alphabeta_search(
    state: GameState,
    depth: int,
    evaluation_function: EvaluationFunction,
    rng: random.Random,
    *,
    on_expand: Callable[[], None] | None = None,
) -> tuple[TeamAction, TeamAction, float]:
    """
    Depth-limited alpha-beta search: same moves as minimax, fewer node expansions.

    Same ply structure as `minimax_search`, with alpha/beta bounds threaded through recursion.

    Tips:
    - Prune the rival (MIN) loop when `score < alpha`; prune Colombia (MAX) when `score > beta`.
    - Use strict inequality — do not prune on equality.
    - Pass updated `alpha` / `beta` into each recursive `ply` call.
    """

    def ply(
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[TeamAction | None, TeamAction | None, float]:
        if on_expand is not None:
            on_expand()

        if is_cutoff(state, depth):
            return None, None, evaluation_function(state)


        """
        Exactamente el mismo codigo de minimax pero aqui se implementa 
        poda alfa-beta evitando evaluar algunas ramas
        """

        colombia_actions = legal_actions(state, Team.COLOMBIA)
        rival_actions = legal_actions(state, Team.RIVAL)

        posibles_escenarios = []
        for colombia_action in colombia_actions:
            respuestas_rival = []

            for rival_action in rival_actions:
                successor = step(state, colombia_action, rival_action)
                _, _, value = ply(successor, depth - 1, alpha, beta)
                respuestas_rival.append((rival_action, value))

                # Aquí se implementa la poda alfa donde alpha siempre maximiza y evita ramas innecesarias
                if value < alpha:
                    break

            worst_rival_action, worst_value = min(respuestas_rival, key=lambda item: item[1])
            posibles_escenarios.append((colombia_action, worst_rival_action, worst_value))

            # Y aquí va la otra que siempre minimiza (Beta)
            if worst_value > beta:
                break

            alpha = max(alpha, worst_value)

        best_colombia_action, best_rival_action, best_value = max(posibles_escenarios, key=lambda item: item[2])
        return best_colombia_action, best_rival_action, best_value

    return finish_search_root(*ply(state, depth, float("-inf"), float("inf")))