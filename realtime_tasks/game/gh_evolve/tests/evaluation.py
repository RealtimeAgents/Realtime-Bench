"""Success criterion for gh_evolve."""
from realtime_tasks.game.verifier import score_target
SCORE_FIELD = 'game_state.evolution_upgrades'
TARGET = 4
def evaluate(state, *, saved_start=None, saved_best=None):
    return score_target(state, score_field=SCORE_FIELD, target_score=TARGET,
                        saved_start=saved_start, saved_best=saved_best)
