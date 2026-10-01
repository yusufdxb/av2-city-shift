import numpy as np

from cityshift.route_censor import N_LOG, score_on_route, window_cutoff


def test_window_cutoff_is_last_step_all_arms_are_on_route():
    assert window_cutoff(np.array([[70, 90], [110, 110], [50, 120]])).tolist() == [69, 109, 49]


def test_score_on_route_counts_only_segments_inside_the_cutoff():
    exec_decel = np.zeros((4, 6))
    exec_decel[:, 2] = -5.0  # replan 69, segment ends at step 79
    log_decel = np.zeros((4, N_LOG))
    log_decel[3, 40] = -4.5  # human window starting at 89 ends at 99
    fault = np.array([-1, 85, 85, -1])
    r = score_on_route(exec_decel, log_decel, fault, np.array([78, 79, 109, 109]))
    assert r["planner_hard_brake"].tolist() == [False, True, True, True]
    assert r["logged_hard_brake"].tolist() == [False, False, False, True]
    assert r["unnecessary_hard_brake"].tolist() == [False, True, True, False]
    assert r["collision"].tolist() == [False, False, True, False]
    assert r["scored_replans"].tolist() == [2, 3, 6, 6]
