import numpy as np


class RuleCoveragePolicy:
    """最近可见目标、按距离让位、角点饱和的无状态规则策略。"""

    def reset(self, context):
        return None

    @staticmethod
    def _corner(vector):
        if not np.any(np.abs(vector) > 0.0):
            return np.zeros(2, dtype=np.float32)
        return np.where(vector >= 0.0, 1.0, -1.0).astype(np.float32)

    def act(self, observation):
        targets = observation["targets"]
        valid_targets = observation["target_exists"] & observation["target_visible"]
        target_indices = np.flatnonzero(valid_targets)
        if target_indices.size == 0:
            return np.zeros(2, dtype=np.float32)

        target_vectors = targets[target_indices, :2].astype(np.float64)
        own_distances = np.linalg.norm(target_vectors, axis=1)
        order = np.argsort(own_distances, kind="stable")

        peers = observation["peers"]
        valid_peers = observation["peer_exists"] & observation["peer_visible"]
        peer_indices = np.flatnonzero(valid_peers)
        peer_vectors = peers[peer_indices, :2].astype(np.float64)
        own_index = int(observation["agent_index"])

        for ordered_index in order:
            target_index = int(target_indices[ordered_index])
            target_vector = targets[target_index, :2].astype(np.float64)
            own_distance = float(own_distances[ordered_index])

            beaten = False
            for peer_index, peer_vector in zip(peer_indices, peer_vectors):
                peer_distance = float(np.linalg.norm(target_vector - peer_vector))
                if peer_distance < own_distance or (
                    peer_distance == own_distance and int(peer_index) < own_index
                ):
                    beaten = True
                    break

            if not beaten:
                return self._corner(target_vector)

        return np.zeros(2, dtype=np.float32)

    def close(self):
        return None


def build_policy(context):
    return RuleCoveragePolicy()
