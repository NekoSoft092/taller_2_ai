from __future__ import annotations

from engine.geometry import (
    manhattan_distance,
    nearest_to,
    path_blocked,
    segment_intercepted,
    sorted_by_manhattan_distance,
    within_manhattan_range,
)
from engine.model import MatchOutcome, Position, Team
from engine.rules import compute_shot_plan
from engine.state import GameState

# --- 1: Instrucción Terminal Scores ---

PUNTAJE_VICTORIA = 1000.0
PUNTAJE_DERROTA = -1000.0
PUNTAJE_EMPATE = 0

# --- 2: Puntajes ofensivos ---

PUNTAJE_AVANCE=3.0
PUNTAJE_DISTANCIA_META=4.0
PUNTAJE_DISPARO_COL=6.0
PUNTAJE_PASE_COL=3.0
PUNTAJE_APOYO=1.5
PUNTAJE_PRESION=1.0 #añadida tras refinación con conversación con la IA, usando el prompt: Qué métricas relevantes al mundo del problema pueden estar haciendo falta en el modelo de la función de evaliación?
PUNTAJE_DISPERSION=0.5 #añadida tras refinación con conversación con la IA

# --- 3: Puntajes defensivos ---

PUNTAJE_PELIGRO=5.0
PUNTAJE_DEFENSA=2.5
PUNTAJE_LADO_GOAL=1.5
PUNTAJE_MARCAJE=1.5 #añadida tras refinación con conversación con la IA
PUNTAJE_DISPARO_RIVAL=4.0
PUNTAJE_PASE_RIVAL=1.5
PUNTAJE_LINEA_CERRADA=2.0

# --- 3: Puntajes defensivos ---

PENALIZACION_TURNO=0.05

def evaluation_function(state: GameState) -> float:
    """
    Estimate how good state is for Colombia (higher = better).

    Colombia attacks toward +x; the rival toward -x. Terminal scores must
    dominate: WIN/LOSS first, then possession-specific terms.

    With the ball (offense):
    - ball[0] / scenario.width and distance to the nearest rival-goal cell.
    - compute_shot_plan(state, Team.COLOMBIA): shoot if target is a goal mouth;
      prefer passes with Team.is_ahead_on_attack_axis.
    - path_blocked / segment_intercepted on shot and pass lanes.
    - Squad spread and teammates **ahead** of the ball so STOP-heavy lines score low.

    Without the ball (defense):
    - Distance from the ball to our goal mouth (danger).
    - Nearest Colombian to the ball (press).
    - Defenders behind the ball on x (goal-side).
    - Per-rival marking distance; segment_intercepted on rival shot/pass lanes.
    - Width via spread so everyone does not camp on one cell.

    General tips
    - Normalize distances with scenario.width + scenario.height.
    - state.turn / max_turns: small penalty to finish attacks.
    - Tune weights per scenario (1v1 vs 5v2); document choices in your report.

    Returns:
        Scalar utility from Colombia's perspective.
    """
    ### YOUR CODE HERE ###
    # --- SOLUTION START ---
    if state.outcome is MatchOutcome.WIN:
        return PUNTAJE_VICTORIA
    if state.outcome is MatchOutcome.LOSS:
        return PUNTAJE_DERROTA
    if state.outcome is MatchOutcome.DRAW:
        return PUNTAJE_EMPATE
    
    # Se separa la evaluación en dos partes: ofensiva y defensiva, dependiendo de si Colombia tiene la posesión del balón o no, cambio realizado tras refinación.
 
    if state.has_ball(Team.COLOMBIA):
        puntaje = puntaje_ofensiva(state)
    else:
        puntaje = puntaje_defensa(state)
 
    max_turns = max(state.scenario.max_turns, 1)
    puntaje -= PENALIZACION_TURNO * (state.turn / max_turns)
 
    return puntaje
    # --- SOLUTION END ---

def puntaje_ofensiva(state: GameState) -> float:

    escenario = state.scenario
    norm = max(escenario.width + escenario.height, 1)
    balon = state.get_ball_position()
    duenio_idx = state.ball_owner
    colombia = state.get_team_positions(Team.COLOMBIA)
    rival = state.get_team_positions(Team.RIVAL)
 
    puntaje = 0.0
    
    avance = balon[0] / max(escenario.width - 1, 1)
    puntaje += PUNTAJE_AVANCE * avance
    
    descarte, goal_cercano = nearest_to(balon, escenario.rival_goal)
    distancia_goal = manhattan_distance(balon, goal_cercano) / norm
    puntaje += PUNTAJE_DISTANCIA_META * (1.0 - distancia_goal)
    
    plan = compute_shot_plan(state, Team.COLOMBIA)
    if plan is not None:
        descarte, target = plan
        if target in escenario.rival_goal:
            puntaje += PUNTAJE_DISPARO_COL
        else:
            puntaje += PUNTAJE_PASE_COL 
    
    apoyo = sum(
        1
        for idx, pos in enumerate(colombia)
        if idx != duenio_idx
        and Team.COLOMBIA.is_ahead_on_attack_axis(balon, pos, inclusive=False)
    )
    puntaje += PUNTAJE_APOYO * apoyo
    
    #añadida tras refinación con conversación con la IA, medir presión como un indicador secundario pero relevante para evitar el avance a zonas con muchos rivales cercanos.
    
    if rival:
        descarte, nearest_rival = nearest_to(balon, rival)
        presion = manhattan_distance(balon, nearest_rival) / norm
        puntaje += PUNTAJE_PRESION * presion
    
    #añadida tras refinación con conversación con la IA, medir dispersión como un indicador secundario pero relevante para evitar que todos los jugadores se concentren en una misma columna, lo que podría facilitar la defensa del rival.
        
    columnas_unicas = len({pos[0] for pos in colombia})
    puntaje += PUNTAJE_DISPERSION * (columnas_unicas / max(len(colombia), 1))
    
    return puntaje

def puntaje_defensa(state: GameState) -> float:
    """Score a state where the rival currently holds the ball."""
    escenario = state.scenario
    norm = max(escenario.width + escenario.height, 1)
    balon = state.get_ball_position()
    colombia = state.get_team_positions(Team.COLOMBIA)
    rival = state.get_team_positions(Team.RIVAL)
 
    puntaje = 0.0
 
    descarte, goal_cercano = nearest_to(balon, escenario.own_goal)
    peligro = manhattan_distance(balon, goal_cercano) / norm
    puntaje += PUNTAJE_PELIGRO * peligro  
 
    if colombia:
        descarte, colombiano_cercano = nearest_to(balon, colombia)
        distancia_defensa = manhattan_distance(balon, colombiano_cercano) / norm
        puntaje += PUNTAJE_DEFENSA * (1.0 - distancia_defensa)
 
    propio_goal_x = next(iter(escenario.own_goal))[0]
    if colombia:
        goal_side = sum(
            1
            for pos in colombia
            if abs(pos[0] - propio_goal_x) <= abs(balon[0] - propio_goal_x)
        )
        puntaje += PUNTAJE_LADO_GOAL * (goal_side / len(colombia))
        
    #añadida tras refinación con conversación con la IA, medir qué tan cerca están los jugadores colombianos de los rivales para evitar que estos tengan oportunidades de disparo o pase, incentivando la marca efectiva.
    
    if colombia and rival:
        costo_marcado = 0.0
        for rival_pos in rival:
            closest = sorted_by_manhattan_distance(rival_pos, colombia)[0]
            costo_marcado += manhattan_distance(rival_pos, closest) / norm
        puntaje -= PUNTAJE_MARCAJE * (costo_marcado / len(rival))
 
    plan = compute_shot_plan(state, Team.RIVAL)
    if plan is None:
        puntaje += PUNTAJE_LINEA_CERRADA
    else:
        descarte, target = plan
        if target in escenario.own_goal:
            puntaje -= PUNTAJE_DISPARO_RIVAL
        else:
            puntaje -= PUNTAJE_PASE_RIVAL
 
    if within_manhattan_range(balon, goal_cercano, escenario.max_shot_distance):
        if plan is not None and plan[1] in escenario.own_goal:
            puntaje -= PUNTAJE_DISPARO_RIVAL * 0.5
 
    return puntaje