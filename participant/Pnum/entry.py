import json
from pathlib import Path
import numpy as np


class TemplatePolicy:
    def __init__(self, artifact_dir: Path):
        cfg_file = artifact_dir / "policy-config.json"
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.action = np.asarray(data.get("action", [0.0, 0.0]), dtype=np.float32)
        self.agent_index = 0

    def reset(self, context):
        self.agent_index = context.agent_index

    def act(self, observation):
        return self.action.copy()

    def close(self):
        pass


def build_policy(context):
    return TemplatePolicy(context.artifact_dir)
