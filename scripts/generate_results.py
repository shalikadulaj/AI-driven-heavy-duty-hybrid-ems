"""Generate the documented synthetic Double-DQN training artifacts.

These are software-harness results. They are not measurements from a truck and
must not be used as evidence of fuel economy, safety, or battery life.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from global_ems.rl import SyntheticEMSEnv, train_double_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    agent, returns = train_double_dqn(episodes=args.episodes, seed=args.seed)
    evaluation_returns: list[float] = []
    evaluation_fallbacks: list[int] = []
    for seed in range(args.seed, args.seed + 10):
        env = SyntheticEMSEnv(seed=seed)
        observation = env.reset()
        total = 0.0
        fallbacks = 0
        done = False
        while not done:
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            total += reward
            fallbacks += int(info["battery_temp_c"] >= 45.0)
        evaluation_returns.append(total)
        evaluation_fallbacks.append(fallbacks)

    with (args.output / "double_dqn_training_returns.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "return"])
        writer.writerows((index, value) for index, value in enumerate(returns, start=1))
    summary = {
        "algorithm": "constrained Double-DQN",
        "environment": "SyntheticEMSEnv",
        "training_episodes": args.episodes,
        "training_seed": args.seed,
        "evaluation_episodes": len(evaluation_returns),
        "mean_evaluation_return": float(np.mean(evaluation_returns)),
        "std_evaluation_return": float(np.std(evaluation_returns, ddof=1)),
        "mean_fallback_count": float(np.mean(evaluation_fallbacks)),
        "interpretation": "software-harness result only; not truck measurement",
    }
    (args.output / "double_dqn_evaluation_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(np.arange(1, len(returns) + 1), returns, alpha=0.35, label="Episode return")
    window = max(1, min(20, len(returns) // 5))
    moving_average = np.convolve(returns, np.ones(window) / window, mode="valid")
    axis.plot(np.arange(window, len(returns) + 1), moving_average, label=f"{window}-episode moving average")
    axis.set_title("Constrained Double-DQN EMS training (synthetic environment)")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Multi-objective return (higher is better)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output / "double_dqn_training_curve.png", dpi=180)
    plt.close(figure)
    print(summary)


if __name__ == "__main__":
    main()
