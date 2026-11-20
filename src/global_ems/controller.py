"""Constraint-first EMS controller and deterministic safety shield.

This is intentionally not presented as a validated direct-actuation controller.
It is a transparent baseline and an interface for a future SAC/TD3/PPO policy.
The learned policy must propose a torque; this module clamps or rejects it
before any lower-level actuator interface can consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .model import (
    ControllerCommand,
    EMSMeasurement,
    ObjectiveWeights,
    SafetyLimits,
)


@dataclass(frozen=True)
class CandidateScore:
    torque_nm: float
    cost: float


class ConstrainedEMSController:
    """Choose rear e-axle torque using balanced multi-objective screening."""

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        weights: ObjectiveWeights | None = None,
        candidate_count: int = 25,
    ) -> None:
        self.limits = limits or SafetyLimits()
        self.weights = weights or ObjectiveWeights()
        self.weights.validate()
        if candidate_count < 3:
            raise ValueError("candidate_count must be at least 3")
        self.candidate_count = candidate_count

    def propose(self, measurement: EMSMeasurement, policy_torque_nm: float | None = None) -> ControllerCommand:
        """Return a bounded command; policy_torque_nm is only a proposal."""
        measurement.validate()
        if self._unsafe_measurement(measurement):
            return self._fallback(measurement, "thermal or SoC safety limit")

        candidates = self._candidate_torques(measurement, policy_torque_nm)
        best = min(
            (CandidateScore(torque, self._cost(measurement, torque)) for torque in candidates),
            key=lambda score: score.cost,
        )
        regenerative = max(0.0, -best.torque_nm)
        friction_request = self._friction_brake_request(measurement, regenerative)
        return ControllerCommand(
            motor_torque_nm=best.torque_nm,
            regenerative_torque_nm=regenerative,
            friction_brake_request_pct=friction_request,
            fallback_active=False,
            reason="balanced constrained candidate selected",
        )

    def _candidate_torques(
        self,
        measurement: EMSMeasurement,
        policy_torque_nm: float | None,
    ) -> tuple[float, ...]:
        lower = -self.limits.maximum_regenerative_torque_nm
        upper = self.limits.maximum_motor_torque_nm
        if measurement.battery_soc_pct <= self.limits.minimum_soc_pct:
            lower = 0.0
        if measurement.battery_soc_pct >= self.limits.maximum_soc_pct:
            lower = 0.0
        if measurement.brake_pct <= 0.0:
            lower = 0.0
        rate = self.limits.maximum_motor_torque_rate_nm_s
        lower = max(lower, measurement.previous_motor_torque_nm - rate)
        upper = min(upper, measurement.previous_motor_torque_nm + rate)
        if lower > upper:
            return (0.0,)
        step = (upper - lower) / (self.candidate_count - 1)
        grid = tuple(lower + index * step for index in range(self.candidate_count))
        if policy_torque_nm is None:
            return grid
        return tuple(sorted(set(grid + (min(upper, max(lower, policy_torque_nm)),))))

    def _cost(self, measurement: EMSMeasurement, torque_nm: float) -> float:
        limits = self.limits
        weights = self.weights
        normalized_torque = abs(torque_nm) / max(limits.maximum_motor_torque_nm, 1.0)
        rate = abs(torque_nm - measurement.previous_motor_torque_nm)
        comfort = rate / max(limits.maximum_motor_torque_rate_nm_s, 1.0)
        high_soc = max(0.0, measurement.battery_soc_pct - 75.0) / 25.0
        low_soc = max(0.0, 30.0 - measurement.battery_soc_pct) / 30.0
        thermal = (
            max(0.0, measurement.battery_temperature_c - 35.0) / 10.0
            + max(0.0, measurement.inverter_temperature_c - 70.0) / 30.0
            + max(0.0, measurement.motor_temperature_c - 120.0) / 30.0
        )
        motor_wear = normalized_torque**2 + (rate / max(limits.maximum_motor_torque_rate_nm_s, 1.0)) ** 2
        electrical_energy = normalized_torque * (1.0 + high_soc + low_soc)
        friction_wear = 0.0 if torque_nm < 0.0 else measurement.brake_pct / 100.0
        safety = 100.0 * thermal + 100.0 * max(0.0, -torque_nm - limits.maximum_regenerative_torque_nm)
        fuel_proxy = max(0.0, measurement.wheel_torque_request_nm - torque_nm) / max(
            abs(measurement.wheel_torque_request_nm), 1.0
        )
        return (
            weights.fuel * fuel_proxy
            + weights.comfort * comfort
            + weights.safety * safety
            + weights.battery_energy * electrical_energy
            + weights.battery_health * (high_soc + low_soc + thermal)
            + weights.motor_durability * motor_wear
            + weights.mechanical_brake_wear * friction_wear
        )

    def _unsafe_measurement(self, measurement: EMSMeasurement) -> bool:
        return (
            measurement.battery_temperature_c >= self.limits.maximum_battery_temperature_c
            or measurement.inverter_temperature_c >= self.limits.maximum_inverter_temperature_c
            or measurement.motor_temperature_c >= self.limits.maximum_motor_temperature_c
            or measurement.battery_soc_pct <= 5.0
            or measurement.battery_soc_pct >= 98.0
        )

    def _friction_brake_request(self, measurement: EMSMeasurement, regenerative_torque_nm: float) -> float:
        if measurement.brake_pct <= 0.0:
            return 0.0
        regen_fraction = min(1.0, regenerative_torque_nm / max(self.limits.maximum_regenerative_torque_nm, 1.0))
        return max(0.0, measurement.brake_pct * (1.0 - regen_fraction))

    def _fallback(self, measurement: EMSMeasurement, reason: str) -> ControllerCommand:
        return ControllerCommand(
            motor_torque_nm=0.0,
            regenerative_torque_nm=0.0,
            friction_brake_request_pct=measurement.brake_pct,
            fallback_active=True,
            reason=reason,
        )
