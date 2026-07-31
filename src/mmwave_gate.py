"""
mmWave Physical Precondition Gate (STEP 2 + STEP 3)

This module implements a gating layer that checks whether an IMU-predicted
gesture is physically plausible given what the mmWave radar sees.

Design (per user spec):
  - NOT a classifier — never predicts a gesture label.
  - Returns True/False: does the current mmWave window satisfy the physical
    preconditions for the IMU-predicted gesture?
  - Thresholds are derived empirically from the 5th percentile of true-positive
    windows across all labeled sessions (see analyze script output).
    
Usage:
    from src.mmwave_gate import mmwave_confirms

    if mmwave_confirms(imu_predicted_label, mmwave_window_features):
        emit(imu_predicted_label, imu_confidence)
    else:
        suppress()   # hold last confirmed or emit "no gesture"
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Feature indices into the 13-element mmWave window-feature vector
# produced by realtime_demo._compute_features(readings, "mmwave")
#
# The 13-vector = [6 means, 6 stds, 1 path_length]
# where the 6 per-frame features are:
#   [num_points, mean_x, std_x, mean_y, std_y, distance_from_origin]
# ---------------------------------------------------------------------------

# Indices (0-5: mean, 6-11: std, 12: path_length)
I_MEAN_NUM_POINTS = 0
I_MEAN_X = 1
I_STD_X = 2
I_MEAN_Y = 3
I_STD_Y = 4
I_MEAN_DIST = 5
I_STD_NUM_POINTS = 6
I_STD_MEAN_X = 7
I_STD_STD_X = 8
I_STD_MEAN_Y = 9
I_STD_STD_Y = 10
I_STD_DIST = 11
I_PATH_LENGTH = 12


# ---------------------------------------------------------------------------
# Per-gesture physical precondition rules.
#
# Each rule is a list of (feature_index, comparator, threshold) tuples.
# ALL conditions must be met for the gate to pass (AND logic).
# Comparator is one of: "ge" (≥), "le" (≤), "abs_ge" (abs ≥).
#
# Thresholds are derived from the P5 (5th percentile) of true-positive
# windows from all labeled mmWave sessions.  This means we will
# incorrectly veto ~5% of true gestures, but will reject almost all
# motion that doesn't meet the gesture's physical minimum.
#
# These are conservative — the intention is to veto clearly impossible
# predictions (e.g., "boxing" when the hand is not in the radar FOV).
# ---------------------------------------------------------------------------

# Universal minimum — only applies to ZONE 1 gestures (ge rules).
# Zone 2 gestures (le rules) bypass this check because seeing few or
# zero points is expected when the arm is near the body.
_UNIVERSAL_MIN_POINTS = 5.0

# Zone 2 gestures use le (≤) rules — the arm must NOT be extended
# into the mmWave FOV.  These skip the freshness + universal checks.
_ZONE_2_GESTURES: set[str] = {
    "t-arm", "clockwise", "anticlockwise", "bye-bye", "clapping",
}

GESTURE_RULES: dict[str, list[tuple[int, str, float]]] = {
    # ──────────────────────────────────────────────────────────────────
    # ZONE 1 — mmWave FOV REQUIRED  (ge rules, arm must be in radar)
    # ──────────────────────────────────────────────────────────────────

    # ── Boxing ───────────────────────────────────────────────────────
    "one-arm-boxing": [
        (I_MEAN_NUM_POINTS, "ge", 7.0),
        (I_MEAN_DIST, "ge", 0.8),
        (I_PATH_LENGTH, "ge", 0.8),
        (I_STD_MEAN_X, "ge", 0.04),
    ],
    "two-arm-boxing": [
        (I_MEAN_NUM_POINTS, "ge", 7.0),
        (I_MEAN_DIST, "ge", 0.8),
        (I_PATH_LENGTH, "ge", 0.8),
        (I_STD_MEAN_X, "ge", 0.04),
    ],

    # ── Push / Pull ──────────────────────────────────────────────────
    "push": [
        (I_MEAN_NUM_POINTS, "ge", 5.0),
        (I_MEAN_DIST, "ge", 0.5),
        (I_PATH_LENGTH, "ge", 0.8),
        (I_STD_DIST, "ge", 0.05),
    ],
    "pull": [
        (I_MEAN_NUM_POINTS, "ge", 5.0),
        (I_MEAN_DIST, "ge", 0.5),
        (I_PATH_LENGTH, "ge", 0.8),
        (I_STD_DIST, "ge", 0.04),
    ],

    # ── Soli ─────────────────────────────────────────────────────────
    "soli": [
        (I_MEAN_NUM_POINTS, "ge", 5.0),
        (I_MEAN_DIST, "ge", 0.3),
    ],

    # ── Making fist open / Palm up-down ──────────────────────────────
    "making-fist-open": [
        (I_MEAN_NUM_POINTS, "ge", 5.0),
        (I_MEAN_DIST, "ge", 0.3),
        (I_PATH_LENGTH, "ge", 0.3),
    ],
    "palm-up-down": [
        (I_MEAN_NUM_POINTS, "ge", 5.0),
        (I_MEAN_DIST, "ge", 0.3),
        (I_PATH_LENGTH, "ge", 0.3),
    ],

    # ── Raise arms (arm in FOV, elevated) ────────────────────────────
    "raise-arms": [
        (I_MEAN_NUM_POINTS, "ge", 5.0),
        (I_MEAN_DIST, "ge", 0.5),
        (I_MEAN_Y, "ge", 0.8),
        (I_PATH_LENGTH, "ge", 0.5),
    ],

    # ──────────────────────────────────────────────────────────────────
    # ZONE 2 — mmWave FOV PROHIBITED  (le rules, arm must NOT be in radar)
    # ──────────────────────────────────────────────────────────────────

    # ── T-arm ────────────────────────────────────────────────────────
    "t-arm": [
        (I_MEAN_NUM_POINTS, "le", 6.0),   # few points = arm near body
        (I_MEAN_DIST,      "le", 0.6),   # close to body
    ],

    # ── Circular gestures (near-body) ────────────────────────────────
    "clockwise": [
        (I_MEAN_NUM_POINTS, "le", 6.0),
        (I_MEAN_DIST,      "le", 0.6),
    ],
    "anticlockwise": [
        (I_MEAN_NUM_POINTS, "le", 6.0),
        (I_MEAN_DIST,      "le", 0.6),
    ],

    # ── Bye-bye / Clapping (near-body) ───────────────────────────────
    "bye-bye": [
        (I_MEAN_NUM_POINTS, "le", 6.0),
        (I_MEAN_DIST,      "le", 0.6),
    ],
    "clapping": [
        (I_MEAN_NUM_POINTS, "le", 6.0),
        (I_MEAN_DIST,      "le", 0.6),
    ],
}

# Gestures with NO rules (bypass the gate entirely):
#   left, right
# These always pass through regardless of what the mmWave sees.
# has_rules_for() returns False, so realtime_demo.py skips them.


# ---------------------------------------------------------------------------
# Confusable gesture pairs (STEP 3)
#
# These pairs require the SAME label from IMU across N consecutive windows
# AND mmwave_confirms == True on at least one of them, before emitting.
#
# Identified from IMU confusion matrix analysis:
#   - push/pull: opposite direction, same arm motion — IMU can confuse
#   - left/right: same
#   - clockwise/anticlockwise: same
#   - one-arm-boxing/two-arm-boxing: same motion, different arm count
#   - raise-arms/t-arm: both arm elevation
#   - making-fist-open/palm-up-down: both hand articulation
# ---------------------------------------------------------------------------

CONFUSABLE_PAIRS: list[tuple[str, str, int]] = [
    ("pull", "push", 2),                      # require 2 consecutive same label
    ("left", "right", 2),
    ("clockwise", "anticlockwise", 2),
    ("one-arm-boxing", "two-arm-boxing", 2),
    ("raise-arms", "t-arm", 2),
    ("making-fist-open", "palm-up-down", 2),
    ("soli", "clapping", 3),                  # both subtle hand motion
    ("bye-bye", "clapping", 3),               # wrist rotation can look similar
]


def _build_confusable_map() -> dict[str, tuple[str, int]]:
    """Build a fast lookup: gesture -> (its pair_label, required_consecutive)."""
    m: dict[str, tuple[str, int]] = {}
    for g1, g2, n in CONFUSABLE_PAIRS:
        m[g1] = (g2, n)
        m[g2] = (g1, n)
    return m


_CONFUSABLE_MAP = _build_confusable_map()


# ---------------------------------------------------------------------------
# Core gating function
# ---------------------------------------------------------------------------

_FEATURE_NAMES = {
    0: "mean_num_points", 1: "mean_x", 2: "std_x",
    3: "mean_y", 4: "std_y", 5: "mean_dist",
    6: "std_num_points", 7: "std_mean_x", 8: "std_std_x",
    9: "std_mean_y", 10: "std_std_y", 11: "std_dist",
    12: "path_length",
}


def mmwave_confirms(
    gesture_label: str,
    mmwave_window_features: list[float],
    last_frame_num_points: float | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Return (True, "") if the mmWave window satisfies physical preconditions.

    Returns (False, "reason") if any precondition fails — the second element
    describes exactly which condition failed and what the actual value was.

    Args:
        gesture_label: The gesture predicted by IMU (e.g. "push").
        mmwave_window_features: 13-element vector from
            _compute_features(mm_readings, "mmwave").
        last_frame_num_points: Num points in the MOST RECENT single frame.
            Used as a freshness check — if the latest frame has 0 points
            the hand just left, veto immediately.
        verbose: If True, also print the reason to stderr.

    Returns:
        (True, "") if ALL preconditions are met.
        (False, "reason") if any precondition fails.
    """
    if len(mmwave_window_features) < 13:
        reason = f"feature vector too short ({len(mmwave_window_features)})"
        if verbose:
            print(f"  [gate] VETO: {reason}")
        return False, reason

    mean_np = mmwave_window_features[I_MEAN_NUM_POINTS]
    is_zone2 = gesture_label in _ZONE_2_GESTURES

    # --- Freshness check: if the MOST RECENT frame has 0 points ---
    if not is_zone2 and last_frame_num_points is not None and last_frame_num_points < 1.0:
        reason = (f"hand just left FOV (last_frame_pts={last_frame_num_points:.0f}, "
                  f"window_mean={mean_np:.1f})")
        if verbose:
            print(f"  [gate] VETO {gesture_label}: {reason}")
        return False, reason

    # --- Universal check: if no hand is visible to the mmWave ---
    if not is_zone2 and mean_np < _UNIVERSAL_MIN_POINTS:
        reason = (f"no hand in mmWave FOV (mean_num_points={mean_np:.1f} "
                  f"< {_UNIVERSAL_MIN_POINTS:.0f})")
        if verbose:
            print(f"  [gate] VETO {gesture_label}: {reason}")
        return False, reason

    # --- Gesture-specific check ---
    rules = GESTURE_RULES.get(gesture_label)
    if rules is None:
        return True, ""

    for idx, comp, threshold in rules:
        val = mmwave_window_features[idx] if idx < len(mmwave_window_features) else 0.0

        if comp == "ge":
            ok = val >= threshold
        elif comp == "le":
            ok = val <= threshold
        elif comp == "abs_ge":
            ok = abs(val) >= threshold
        else:
            ok = True

        if not ok:
            fname = _FEATURE_NAMES.get(idx, f"feat_{idx}")
            reason = f"{fname}={val:.4f} fails {comp}({threshold:.4f})"
            if verbose:
                print(f"  [gate] VETO {gesture_label}: {reason}")
            return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# Double-confirmation helper (STEP 3)
# ---------------------------------------------------------------------------

def needs_double_confirmation(gesture_label: str) -> tuple[str, int] | None:
    """Return (pair_label, required_consecutive_windows) if gesture is confusable.

    Returns None if no double-confirmation is needed for this label.
    """
    return _CONFUSABLE_MAP.get(gesture_label)


def check_double_confirmation(
    gesture_label: str,
    consecutive_count: int,
) -> bool:
    """Check if the same gesture has been predicted for enough consecutive windows.

    Args:
        gesture_label: The current IMU prediction.
        consecutive_count: Number of consecutive windows this label has appeared.

    Returns:
        True if the required consecutive count has been reached.
    """
    pair_info = needs_double_confirmation(gesture_label)
    if pair_info is None:
        return True  # no double-confirmation needed
    _, required = pair_info
    return consecutive_count >= required


# ---------------------------------------------------------------------------
# Convenience: check if mmWave gate can run (at least one rule exists)
# ---------------------------------------------------------------------------

def is_zone2_gesture(gesture_label: str) -> bool:
    """Return True if this gesture is in Zone 2 (le rules, arm NOT in mmWave FOV).

    Zone 2 gestures expect the arm to be near the body (few or zero mmWave
    points).  The freshness check and universal minimum are skipped for these.
    """
    return gesture_label in _ZONE_2_GESTURES


def has_rules_for(gesture_label: str) -> bool:
    """Return True if this gesture has any physical precondition rules."""
    return gesture_label in GESTURE_RULES
