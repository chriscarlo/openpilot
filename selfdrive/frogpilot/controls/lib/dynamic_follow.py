from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW
import numpy as np

def sigmoid(x, L, k, x0):
    epsilon = 1e-6  # Small value to prevent division by zero
    return L / (1 + np.exp(-k * (x - x0)) + epsilon)

def calculate_dynamic_follow(base_follow, v_ego, personality):
    v_ego_mph = v_ego * 2.23694  # Convert m/s to mph

    if personality == "aggressive":
        min_follow = 0.65
        max_follow = 1.65
        L = max_follow - min_follow
        k = 0.075
        x0 = 35
    elif personality == "standard":
        min_follow = 0.6
        max_follow = 1.85
        L = max_follow - min_follow
        k = 0.075
        x0 = 25
    elif personality == "relaxed":
        min_follow = 0.5
        max_follow = 2.05
        L = max_follow - min_follow
        k = 0.075
        x0 = 18
    else:
        # Fallback to original calculation
        return np.clip(base_follow * (v_ego / (60 * 0.44704)), 0.8, 2.5)

    target_follow = min_follow + sigmoid(v_ego_mph, L, k, x0)
    return np.clip(target_follow, min_follow, max_follow)

def get_dynamic_follow(aggressive_follow, standard_follow, relaxed_follow, custom_personalities, personality, v_ego):
    base_follow = get_T_FOLLOW(aggressive_follow, standard_follow, relaxed_follow, custom_personalities, personality)
    return calculate_dynamic_follow(base_follow, v_ego, personality)