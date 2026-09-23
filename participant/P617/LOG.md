# 实验日志

- P903 学习基线：SB3 PPO（训练栈：make_training_env → pettingzoo_env_to_vec_env_v1 →
  concat_vec_envs_v1(num_cpus=0) → FlattenCoverageVecEnv（与评测同源 flatten）→ VecMonitor →
  VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10.0)）。
- 种子分离：训练种子 7101（训练随机种子与总环境步数记入 REPORT.md）；公开评测种子不用于训练采样。
- 导出 policy.npz；导出一致性检查（--check-export，max|Δa| ≤ 1e-5）通过后入包。
- 审计硬拒绝提交内 pickle.load（AT-AUD-02），训练栈归一化统计量由 vecnormalize.pkl 改存 vecnormalize-stats.npz，重跑冒烟管线与导出一致性检查通过。
