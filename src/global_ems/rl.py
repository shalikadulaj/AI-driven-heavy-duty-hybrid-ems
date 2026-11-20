"""Executable constrained Double-DQN EMS prototype.

The neural policy proposes one of eleven normalized rear-e-axle torque actions.
The deterministic :class:`ConstrainedEMSController` remains authoritative:
invalid, unsafe, or excessive proposals are clamped before a command is
returned. ``SyntheticEMSEnv`` is a software test environment only; replace it
with a calibrated plant before drawing engineering conclusions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Deque

import numpy as np

from .controller import ConstrainedEMSController
from .model import EMSMeasurement, ObjectiveWeights


ACTION_FRACTIONS = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
OBSERVATION_SIZE = 17


def measurement_observation(measurement: EMSMeasurement) -> np.ndarray:
    """Normalize measured signals for a neural policy without changing units."""
    values = np.asarray(
        [
            measurement.vehicle_speed_kph / 100.0,
            measurement.vehicle_acceleration_m_s2 / 2.0,
            measurement.engine_speed_rpm / 2500.0,
            measurement.engine_load_pct / 100.0,
            measurement.vehicle_gear / 12.0,
            measurement.accelerator_pct / 100.0,
            measurement.brake_pct / 100.0,
            measurement.battery_soc_pct / 100.0,
            measurement.battery_soh_pct / 100.0,
            measurement.battery_voltage_v / 800.0,
            measurement.battery_current_a / 500.0,
            measurement.battery_temperature_c / 60.0,
            measurement.inverter_temperature_c / 120.0,
            measurement.motor_temperature_c / 180.0,
            measurement.dc_bus_voltage_v / 800.0,
            measurement.wheel_torque_request_nm / 4000.0,
            measurement.previous_motor_torque_nm / 3000.0,
        ],
        dtype=np.float32,
    )
    return np.clip(values, -2.0, 2.0)


class ReplayBuffer:
    """Fixed-size replay memory with deterministic NumPy sampling."""

    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._data: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._data.append((state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self._data) < batch_size:
            raise ValueError("not enough transitions in replay buffer")
        indices = self._rng.choice(len(self._data), size=batch_size, replace=False)
        batch = [self._data[int(index)] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._data)


class MLPQNetwork:
    """Small NumPy MLP used to keep the reference trainer dependency-light."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_size)
        self.weights = [
            rng.normal(0.0, scale, (input_size, hidden_size)).astype(np.float32),
            np.zeros(hidden_size, dtype=np.float32),
            rng.normal(0.0, np.sqrt(2.0 / hidden_size), (hidden_size, hidden_size)).astype(np.float32),
            np.zeros(hidden_size, dtype=np.float32),
            rng.normal(0.0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size)).astype(np.float32),
            np.zeros(output_size, dtype=np.float32),
        ]

    def predict(self, states: np.ndarray) -> np.ndarray:
        hidden1 = np.tanh(states @ self.weights[0] + self.weights[1])
        hidden2 = np.tanh(hidden1 @ self.weights[2] + self.weights[3])
        return hidden2 @ self.weights[4] + self.weights[5]

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
        hidden1 = np.tanh(states @ self.weights[0] + self.weights[1])
        hidden2 = np.tanh(hidden1 @ self.weights[2] + self.weights[3])
        output = hidden2 @ self.weights[4] + self.weights[5]
        selected = output[np.arange(len(actions)), actions]
        error = selected - targets
        absolute_error = np.abs(error)
        loss = float(np.mean(np.where(absolute_error <= 1.0, 0.5 * error**2, absolute_error - 0.5)))
        gradient = np.clip(error, -1.0, 1.0) / max(len(actions), 1)
        output_gradient = np.zeros_like(output)
        output_gradient[np.arange(len(actions)), actions] = gradient
        grad_w5 = hidden2.T @ output_gradient
        grad_b5 = output_gradient.sum(axis=0)
        grad_h2 = output_gradient @ self.weights[4].T
        grad_z2 = grad_h2 * (1.0 - hidden2**2)
        grad_w3 = hidden1.T @ grad_z2
        grad_b3 = grad_z2.sum(axis=0)
        grad_h1 = grad_z2 @ self.weights[2].T
        grad_z1 = grad_h1 * (1.0 - hidden1**2)
        grad_w1 = states.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        gradients = (grad_w1, grad_b1, grad_w3, grad_b3, grad_w5, grad_b5)
        for index, gradient_value in enumerate(gradients):
            self.weights[index] -= learning_rate * np.clip(gradient_value, -5.0, 5.0)
        return loss

    def copy(self) -> "MLPQNetwork":
        clone = MLPQNetwork(self.weights[0].shape[0], self.weights[4].shape[1], self.weights[0].shape[1])
        clone.weights = [weight.copy() for weight in self.weights]
        return clone


@dataclass(frozen=True)
class RLConfig:
    gamma: float = 0.98
    learning_rate: float = 0.001
    batch_size: int = 64
    warmup_steps: int = 256
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000


class ConstrainedDoubleDQN:
    """Double-DQN policy whose action is always passed through the safety shield."""

    def __init__(self, seed: int = 0, config: RLConfig | None = None) -> None:
        self.config = config or RLConfig()
        self.online = MLPQNetwork(OBSERVATION_SIZE, len(ACTION_FRACTIONS), seed=seed)
        self.target = self.online.copy()
        self.replay = ReplayBuffer(seed=seed)
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / max(self.config.epsilon_decay_steps, 1))
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * (1.0 - progress)

    def select_action(self, observation: np.ndarray, explore: bool = False) -> int:
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.integers(len(ACTION_FRACTIONS)))
        return int(np.argmax(self.online.predict(observation.reshape(1, -1))[0]))

    def propose_command(
        self,
        measurement: EMSMeasurement,
        shield: ConstrainedEMSController | None = None,
        explore: bool = False,
    ):
        shield = shield or ConstrainedEMSController()
        action = self.select_action(measurement_observation(measurement), explore=explore)
        torque = float(ACTION_FRACTIONS[action] * shield.limits.maximum_motor_torque_nm)
        return shield.propose(measurement, policy_torque_nm=torque)

    def train_step(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)
        online_next = self.online.predict(next_states)
        next_actions = np.argmax(online_next, axis=1)
        target_next = self.target.predict(next_states)[np.arange(len(next_actions)), next_actions]
        targets = rewards + self.config.gamma * (1.0 - dones) * target_next
        loss = self.online.train_batch(states, actions, targets, self.config.learning_rate)
        if self.steps % self.config.target_update_interval == 0:
            self.target = self.online.copy()
        return loss

    def observe(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float | None:
        self.replay.add(state, action, reward, next_state, done)
        self.steps += 1
        return self.train_step()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, *self.online.weights)
        path.with_suffix(".json").write_text(json.dumps({
            "algorithm": "constrained_double_dqn",
            "observation_size": OBSERVATION_SIZE,
            "action_fractions": ACTION_FRACTIONS.tolist(),
            "steps": self.steps,
            "limitations": "Synthetic environment and placeholder limits; not deployment-ready.",
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ConstrainedDoubleDQN":
        policy = cls()
        archive = np.load(path)
        policy.online.weights = [archive[key] for key in sorted(archive.files, key=lambda value: int(value[4:]))]
        policy.target = policy.online.copy()
        return policy


class SyntheticEMSEnv:
    """Small training environment for software tests, not a calibrated truck."""

    def __init__(self, horizon: int = 120, seed: int = 0, weights: ObjectiveWeights | None = None) -> None:
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.weights = weights or ObjectiveWeights()
        self.controller = ConstrainedEMSController(weights=self.weights)
        self.step_index = 0
        self.speed_kph = 50.0
        self.soc_pct = 60.0
        self.battery_temp_c = 25.0
        self.motor_temp_c = 70.0
        self.inverter_temp_c = 45.0
        self.previous_torque = 0.0

    def reset(self) -> np.ndarray:
        self.step_index = 0
        self.speed_kph = float(self.rng.uniform(30.0, 80.0))
        self.soc_pct = float(self.rng.uniform(45.0, 75.0))
        self.battery_temp_c = float(self.rng.uniform(20.0, 32.0))
        self.motor_temp_c = float(self.rng.uniform(55.0, 85.0))
        self.inverter_temp_c = float(self.rng.uniform(35.0, 55.0))
        self.previous_torque = 0.0
        return measurement_observation(self._measurement())

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        if not 0 <= action < len(ACTION_FRACTIONS):
            raise ValueError("action outside discrete torque action space")
        measurement = self._measurement()
        policy_torque = float(ACTION_FRACTIONS[action] * self.controller.limits.maximum_motor_torque_nm)
        command = self.controller.propose(measurement, policy_torque_nm=policy_torque)
        torque = command.motor_torque_nm
        target_speed = 55.0 + 15.0 * np.sin(self.step_index / 17.0)
        speed_error = abs(target_speed - self.speed_kph)
        self.speed_kph += 0.08 * (target_speed - self.speed_kph) + torque / 30_000.0
        self.soc_pct = float(np.clip(self.soc_pct - max(torque, 0.0) / 3000.0 * 0.16 + max(-torque, 0.0) / 3000.0 * 0.10, 10.0, 95.0))
        heat = abs(torque) / 3000.0 * 0.12
        self.battery_temp_c += heat - 0.03 * (self.battery_temp_c - 25.0)
        self.motor_temp_c += heat * 10.0 - 0.04 * (self.motor_temp_c - 70.0)
        self.inverter_temp_c += heat * 8.0 - 0.05 * (self.inverter_temp_c - 45.0)
        torque_rate = abs(torque - self.previous_torque) / 4000.0
        fuel_cost = max(0.0, measurement.wheel_torque_request_nm - torque) / 2000.0
        health_cost = max(0.0, self.battery_temp_c - 35.0) / 10.0 + max(0.0, self.motor_temp_c - 120.0) / 30.0
        mechanical_brake_cost = command.friction_brake_request_pct / 100.0
        reward = -(
            self.weights.fuel * fuel_cost
            + self.weights.comfort * (speed_error / 20.0 + torque_rate)
            + self.weights.battery_health * (health_cost + abs(self.soc_pct - 60.0) / 60.0)
            + self.weights.motor_durability * torque_rate**2
            + self.weights.mechanical_brake_wear * mechanical_brake_cost
            + self.weights.safety * float(command.fallback_active)
        )
        self.previous_torque = torque
        self.step_index += 1
        done = self.step_index >= self.horizon
        info = {"torque_nm": torque, "soc_pct": self.soc_pct, "battery_temp_c": self.battery_temp_c, "reward": reward}
        return measurement_observation(self._measurement()), float(reward), done, info

    def _measurement(self) -> EMSMeasurement:
        target_speed = 55.0 + 15.0 * np.sin(self.step_index / 17.0)
        brake = float(np.clip((self.speed_kph - target_speed) / 15.0, 0.0, 1.0) * 100.0)
        accelerator = 0.0 if brake > 0.0 else 45.0
        return EMSMeasurement(
            timestamp_s=float(self.step_index),
            vehicle_speed_kph=self.speed_kph,
            vehicle_acceleration_m_s2=0.0,
            engine_speed_rpm=1200.0 + self.speed_kph * 5.0,
            engine_load_pct=50.0,
            vehicle_gear=5,
            accelerator_pct=accelerator,
            brake_pct=brake,
            battery_soc_pct=self.soc_pct,
            battery_soh_pct=96.0,
            battery_voltage_v=650.0,
            battery_current_a=0.0,
            battery_temperature_c=self.battery_temp_c,
            inverter_temperature_c=self.inverter_temp_c,
            motor_temperature_c=self.motor_temp_c,
            dc_bus_voltage_v=650.0,
            wheel_torque_request_nm=1200.0 + 400.0 * np.sin(self.step_index / 9.0),
            previous_motor_torque_nm=self.previous_torque,
        )


def train_double_dqn(episodes: int = 20, seed: int = 0) -> tuple[ConstrainedDoubleDQN, list[float]]:
    """Train and return a reproducible software-level Double-DQN policy."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    agent = ConstrainedDoubleDQN(seed=seed, config=RLConfig(batch_size=16, warmup_steps=32))
    returns: list[float] = []
    for episode in range(episodes):
        env = SyntheticEMSEnv(seed=seed + episode)
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            total += reward
        returns.append(total)
    return agent, returns
