# 策略研究报告（参赛模板 · 学习基线示例）

## 训练方案
stable-baselines3 PPO，MlpPolicy [64,64]，观测为与评测同源的 104 维 flatten
（coverage_bench.spaces.flatten_observation），VecNormalize 仅归一化观测。
训练环境：coverage_bench.envs.make_training_env，public suite 任务配置（case basic-0），
独立训练种子序列（默认 7101），公开评测种子不用于训练采样。

## 训练成本
- 总环境步数：24576
- 训练随机种子：7101
- wall time：9.0 秒
- CPU 型号：Windows-11-10.0.26200-SP0
- 学习曲线：training_curve.csv（每曲线点记录总步数与最近 100 回合平均奖励），
  冒烟曲线摘录：首行 total_steps=6144, mean_ep_reward_last100=0.560000；末行 total_steps=24576, mean_ep_reward_last100=0.260000

## 模型选择依据
取训练结束时最终模型（固定策略，不做早停或 checkpoint 挑选）。

## 导出一致性
tools/export_learning_baseline.py 导出 policy.npz（权重 + 归一化统计量）；
train.py --check-export 对 256 个真实环境观测比较 SB3 前向与 numpy 前向的确定性动作，
覆盖观测预处理、归一化、前向、动作裁剪与 dtype 转换，max|Δa| 实测 8.628e-08（容差 1e-5）。

## 定位
给定训练预算下的学习效果与成本参考（spec BD-04）。学习策略未超过 P902 规则基线时，
如实记录本报告数字，并结合学习曲线检查训练适配、训练预算与奖励设计。
