import numpy as np

from entry import RuleCoveragePolicy, build_policy


CAPACITY = 8


def make_observation(*, agent_index=0, targets=None, peers=None):
    target_rows = np.zeros((CAPACITY, 3), dtype=np.float32)
    target_exists = np.zeros(CAPACITY, dtype=np.bool_)
    target_visible = np.zeros(CAPACITY, dtype=np.bool_)
    for index, vector, exists, visible in targets or []:
        target_rows[index, :2] = np.asarray(vector, dtype=np.float32)
        target_rows[index, 2] = np.float32(0.15)
        target_exists[index] = exists
        target_visible[index] = visible

    peer_rows = np.zeros((CAPACITY, 5), dtype=np.float32)
    peer_exists = np.zeros(CAPACITY, dtype=np.bool_)
    peer_visible = np.zeros(CAPACITY, dtype=np.bool_)
    for index, vector, exists, visible in peers or []:
        peer_rows[index, :2] = np.asarray(vector, dtype=np.float32)
        peer_rows[index, 4] = np.float32(0.05)
        peer_exists[index] = exists
        peer_visible[index] = visible

    return {
        "self_state": np.zeros(5, dtype=np.float32),
        "peers": peer_rows,
        "peer_exists": peer_exists,
        "peer_visible": peer_visible,
        "targets": target_rows,
        "target_exists": target_exists,
        "target_visible": target_visible,
        "agent_index": np.int64(agent_index),
        "step_index": np.int64(0),
        "time_remaining": np.float32(1.0),
    }


def assert_action(actual, expected):
    assert actual.shape == (2,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))


def test_build_policy_and_lifecycle_are_stateless():
    policy = build_policy(None)
    assert isinstance(policy, RuleCoveragePolicy)
    assert policy.reset(None) is None
    assert policy.close() is None


def test_no_valid_visible_target_returns_zero_and_masks_empty_slots():
    observation = make_observation(
        targets=[
            (0, (0.1, 0.1), False, True),
            (7, (-0.1, -0.1), True, False),
        ]
    )
    assert_action(RuleCoveragePolicy().act(observation), (0.0, 0.0))


def test_nearest_valid_target_uses_corner_saturation_and_zero_component_is_positive():
    observation = make_observation(
        targets=[
            (1, (-0.4, -0.1), True, True),
            (6, (0.0, 0.2), True, True),
        ]
    )
    assert_action(RuleCoveragePolicy().act(observation), (1.0, 1.0))


def test_exact_zero_target_vector_returns_zero():
    observation = make_observation(
        targets=[(4, (0.0, 0.0), True, True)]
    )
    assert_action(RuleCoveragePolicy().act(observation), (0.0, 0.0))


def test_farther_agent_defers_first_target_and_selects_next_target():
    observation = make_observation(
        agent_index=0,
        targets=[
            (0, (0.4, 0.0), True, True),
            (5, (-0.5, 0.1), True, True),
        ],
        peers=[(1, (0.3, 0.0), True, True)],
    )
    assert_action(RuleCoveragePolicy().act(observation), (-1.0, 1.0))


def test_all_targets_deferred_returns_zero():
    observation = make_observation(
        agent_index=2,
        targets=[(3, (0.4, 0.0), True, True)],
        peers=[(1, (0.3, 0.0), True, True)],
    )
    assert_action(RuleCoveragePolicy().act(observation), (0.0, 0.0))


def test_exact_distance_tie_is_won_only_by_smaller_agent_index():
    target = [(0, (0.5, 0.0), True, True)]
    peer = [(0, (0.0, 0.0), True, True)]
    assert_action(
        RuleCoveragePolicy().act(
            make_observation(agent_index=1, targets=target, peers=peer)
        ),
        (0.0, 0.0),
    )

    higher_index_peer = [(1, (0.0, 0.0), True, True)]
    assert_action(
        RuleCoveragePolicy().act(
            make_observation(agent_index=0, targets=target, peers=higher_index_peer)
        ),
        (1.0, 1.0),
    )


def test_invisible_or_nonexistent_peer_cannot_force_deferral():
    target = [(2, (0.4, -0.2), True, True)]
    peers = [
        (1, (0.3, -0.2), False, True),
        (6, (0.3, -0.2), True, False),
    ]
    assert_action(
        RuleCoveragePolicy().act(
            make_observation(agent_index=0, targets=target, peers=peers)
        ),
        (1.0, -1.0),
    )
