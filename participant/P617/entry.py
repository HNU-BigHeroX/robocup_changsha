"""P617 规则策略：贪心目标分配 + 目标追踪（速度估计 + 提前量拦截）+ 避撞。

设计要点（参数与推导见代码注释）：
1. 分工：所有机器人用同一套确定性贪心算法（按编号顺序，各认领最近的未分配目标），
   保证覆盖不重复，同时避免"机器人 i 被派到离自己很远的固定目标"的几何浪费。
2. 追踪：用绝对位置重建目标的真实速度，预测若干步后的位置，朝预测点走。
3. 控制：把"期望速度"经一步死区控制换算成驱动力（动作），并加避撞排斥。

不加载任何模型文件，纯 NumPy，符合评测环境（无 PyTorch）要求。
"""
import numpy as np


class RulePolicy:
    """单机器人的规则策略实例。每回合 reset 清空记忆，act 输出 2 维驱动力。"""

    # ---- 可调参数 ----
    KP = 3.0          # 速度环比例增益：期望速度 = KP * 位置误差
    V_MAX = 0.5      # 期望速度上限（略高于物理稳态 0.4，让死区控制在贴住目标前才减速）
    LEAD_STEPS = 3.0  # 提前量：预测目标 LEAD_STEPS 步之后的位置
    AVOID_RADIUS = 0.3  # 队友进入该距离开始避撞
    AVOID_GAIN = 0.6    # 避撞排斥强度（速度空间）

    def __init__(self, context):
        self._agent_index = None
        self._num_agents = None
        self._num_targets = None
        self._dt = None
        self._damping = None
        self._deadbeat_gain = None  # mass / (drive_force * dt)

    def reset(self, context):
        M = context.num_targets
        self._agent_index = int(context.agent_index)
        self._num_agents = int(context.num_agents)
        self._num_targets = int(M)

        task = context.task
        self._dt = float(task.dt)
        self._damping = float(task.damping)
        # 死区控制：v_next = (1-damping)*v + (drive_force*act/mass)*dt
        # 令 v_next = v_desired，反解 act = mass/(drive_force*dt) * (v_desired - (1-damping)*v)
        self._deadbeat_gain = float(task.robot_mass) / (float(task.drive_force) * self._dt)

        # 目标状态：绝对位置、速度估计（每步位移）、可见历史
        self._target_pos = np.zeros((M, 2), dtype=np.float64)
        self._target_vel = np.zeros((M, 2), dtype=np.float64)
        self._target_seen = np.zeros(M, dtype=bool)
        self._target_visible_prev = np.zeros(M, dtype=bool)
        self._target_last_seen_step = np.full(M, -1, dtype=np.int64)

        # 队友状态：绝对位置（用于贪心分配）
        self._peer_abs = np.zeros((context.num_agents, 2), dtype=np.float64)
        self._peer_seen = np.zeros(context.num_agents, dtype=bool)
        self._prev_step = None

    def act(self, observation):
        self_pos = observation["self_state"][:2].astype(np.float64)   # 自身绝对位置
        self_vel = observation["self_state"][2:4].astype(np.float64)  # 自身速度
        targets = observation["targets"]                               # [B,3] 相对位置+覆盖半径
        target_visible = observation["target_visible"]                 # [B]
        peers = observation["peers"]                                   # [A,5] 相对位置+相对速度+半径
        peer_visible = observation["peer_visible"]                     # [A]
        step = int(observation["step_index"])

        M = self._num_targets
        # 绝对目标位置 = 自身位置 + 相对位置（机器人自身运动被抵消，只剩目标真实运动）
        abs_targets = self_pos[None, :] + targets[:, :2]

        # 1) 用连续两次可见观测估计目标速度（每步位移）
        if self._prev_step is not None and step > self._prev_step:
            gap = step - self._prev_step
            for j in range(M):
                if target_visible[j] and self._target_visible_prev[j]:
                    vel = (abs_targets[j] - self._target_pos[j]) / gap
                    self._target_vel[j] = 0.7 * self._target_vel[j] + 0.3 * vel

        # 2) 更新目标最近已知位置
        for j in range(M):
            if target_visible[j]:
                self._target_pos[j] = abs_targets[j]
                self._target_seen[j] = True
                self._target_last_seen_step[j] = step

        # 3) 更新队友绝对位置（队友 = 自身 + 相对位置）
        for i in range(self._num_agents):
            if i != self._agent_index and peer_visible[i]:
                self._peer_abs[i] = self_pos + peers[i][:2]
                self._peer_seen[i] = True

        # 4) 贪心分配：机器人按编号顺序认领最近的未分配目标
        tj = self._greedy_assignment(self_pos, step, target_visible)

        # 5) 确定追踪点（含提前量）
        target_point = self._target_point(tj, step, target_visible, self_pos)

        # 6) 期望速度 = KP * 位置误差，限幅
        err = target_point - self_pos
        dist = float(np.linalg.norm(err))
        if dist > 1e-9:
            v_desired = err * self.KP
            speed = float(np.linalg.norm(v_desired))
            if speed > self.V_MAX:
                v_desired = v_desired / speed * self.V_MAX
        else:
            v_desired = np.zeros(2, dtype=np.float64)

        # 7) 避撞：可见队友过近时加排斥
        v_desired = v_desired + self._avoidance(peers, peer_visible)

        # 8) 死区控制换算成驱动力，并夹到 [-1,1]
        action = self._deadbeat_gain * (v_desired - (1.0 - self._damping) * self_vel)
        action = np.clip(action, -1.0, 1.0)

        # 记录状态，供下一步速度估计
        self._target_visible_prev = np.array(target_visible, dtype=bool)
        self._prev_step = step

        return action.astype(np.float32)

    def _known_target_positions(self, step, target_visible):
        """返回各目标当前"已知"绝对位置（可见→直接；见过→外推；否则 None 标记）。"""
        M = self._num_targets
        known_pos = np.zeros((M, 2), dtype=np.float64)
        known = np.zeros(M, dtype=bool)
        for j in range(M):
            if target_visible[j]:
                known_pos[j] = self._target_pos[j]
                known[j] = True
            elif self._target_seen[j]:
                unseen = step - self._target_last_seen_step[j]
                known_pos[j] = self._target_pos[j] + self._target_vel[j] * unseen
                known[j] = True
        return known_pos, known

    def _greedy_assignment(self, self_pos, step, target_visible):
        """确定性贪心分配：机器人按编号顺序，各自认领最近的未分配目标。"""
        N = self._num_agents
        M = self._num_targets
        known_pos, known = self._known_target_positions(step, target_visible)

        # 各机器人当前已知位置（自己→直接；可见队友→直接；见过队友→最后位置）
        robot_pos = np.full((N, 2), np.nan, dtype=np.float64)
        robot_known = np.zeros(N, dtype=bool)
        robot_pos[self._agent_index] = self_pos
        robot_known[self._agent_index] = True
        for i in range(N):
            if i != self._agent_index and self._peer_seen[i]:
                robot_pos[i] = self._peer_abs[i]
                robot_known[i] = True

        taken = [False] * M
        my_target = None
        for i in range(N):
            if not robot_known[i]:
                continue
            best = None
            best_d = 1e18
            for j in range(M):
                if known[j] and not taken[j]:
                    d = float(np.linalg.norm(known_pos[j] - robot_pos[i]))
                    if d < best_d:
                        best_d = d
                        best = j
            if best is not None:
                taken[best] = True
                if i == self._agent_index:
                    my_target = best

        # 兜底：若分配失败（例如一个目标都没见过），追踪最近已知目标
        if my_target is None:
            best_d = 1e18
            for j in range(M):
                if known[j]:
                    d = float(np.linalg.norm(known_pos[j] - self_pos))
                    if d < best_d:
                        best_d = d
                        my_target = j
        return my_target

    def _target_point(self, tj, step, target_visible, self_pos):
        """被分配目标的预测位置（含提前量），用于朝它前进。"""
        if tj is None:
            return self_pos  # 无任何已知目标，原地不动
        M = self._num_targets
        if tj < 0 or tj >= M:
            return self_pos
        if target_visible[tj]:
            return self._target_pos[tj] + self._target_vel[tj] * self.LEAD_STEPS
        if self._target_seen[tj]:
            unseen = step - self._target_last_seen_step[tj]
            return self._target_pos[tj] + self._target_vel[tj] * (self.LEAD_STEPS + unseen)
        return self_pos

    def _avoidance(self, peers, peer_visible):
        """对视野内过近的队友产生排斥速度。"""
        avoidance = np.zeros(2, dtype=np.float64)
        for i in range(self._num_agents):
            if i == self._agent_index:
                continue
            if peer_visible[i]:
                rel = peers[i][:2]  # 相对位置（队友 - 自身）
                d = float(np.linalg.norm(rel))
                if 1e-9 < d < self.AVOID_RADIUS:
                    strength = (self.AVOID_RADIUS - d) / self.AVOID_RADIUS
                    avoidance += (-rel / d) * strength * self.AVOID_GAIN
        return avoidance

    def close(self):
        pass


def build_policy(context):
    """官方加载策略的唯一入口。"""
    return RulePolicy(context)
