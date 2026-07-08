from __future__ import annotations

import random
from collections.abc import Callable

from algorithms.base import EvaluationFunction
from algorithms.base.search import finish_search_root, is_cutoff, legal_actions
from engine.model import Team, TeamAction
from engine.rules import step
from engine.state import GameState


def minimax_search(
    state: GameState,
    depth: int,
    evaluation_function: EvaluationFunction,
    rng: random.Random,
    *,
    on_expand: Callable[[], None] | None = None,
) -> tuple[TeamAction, TeamAction, float]:
    """
    Depth-limited minimax from the root: Colombia MAX action, rival MIN reply, and value.

    Each ply is one simultaneous timestep: Colombia chooses, the rival replies, then `step(...)`.
    Leaf when `depth == 0` or the match is over → `evaluation_function(state)`.

    Tips:
    - Use `legal_actions(state, team)` and `step(state, colombia_action, rival_action)`.
    - Per ply: `max` over Colombia actions of `min` over rival replies.
    - Recurse with `ply(successor, depth - 1)[2]`; only the root needs the returned actions.
    - At the rival layer, `min((action, value), ..., key=lambda item: item[1])` compares by value.
    - Optional: cache `(state, depth)` with a hashable key to speed up search.
    """

    def ply(state: GameState, depth: int) -> tuple[TeamAction | None, TeamAction | None, float]:
        if on_expand is not None:
            on_expand()

        if is_cutoff(state, depth):
            return None, None, evaluation_function(state)

        #Acciones colombia y el rival
        colombia_actions = legal_actions(state, Team.COLOMBIA)
        rival_actions = legal_actions(state, Team.RIVAL)

        # Aquí las mejores opciones que encuentre Colombia
        posibles_escenarios = []

        for colombia_action in colombia_actions:
            
            #respuestas posibles del rival
            respuestas_rival = []
            for rival_action in rival_actions:
                successor = step(state, colombia_action, rival_action)
                _, _, value = ply(successor, depth - 1)
                
                #Se guarda la accion
                respuestas_rival.append((rival_action, value))

            #El rival minimiza
            worst_rival_action, worst_value = min(respuestas_rival, key=lambda item: item[1])
            posibles_escenarios.append((colombia_action, worst_rival_action, worst_value))

        #Colombia maximiza
        best_colombia_action, best_rival_action, best_value = max(posibles_escenarios, key=lambda item: item[2])

        return best_colombia_action, best_rival_action, best_value

    return finish_search_root(*ply(state, depth))
