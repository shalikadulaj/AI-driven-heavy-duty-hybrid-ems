"""Typed measurement, constraint, and objective definitions for the EMS.

The model deliberately separates measured inputs from commands. A future deep
RL policy may propose an action, but the safety filter remains deterministic
and authoritative over that proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EMSMeasurement:
    """One synchronized sample from the truck supervisory network."""

    timestamp_s: float
    vehicle_speed_kph: float
    vehicle_acceleration_m_s2: float
    engine_speed_rpm: float
    engine_load_pct: float
    vehicle_gear: int
    accelerator_pct: float
    brake_pct: float
    battery_soc_pct: float
    battery_soh_pct: float
    battery_voltage_v: float
    battery_current_a: float
    battery_temperature_c: float
    inverter_temperature_c: float
    motor_temperature_c: float
    dc_bus_voltage_v: float
    wheel_torque_request_nm: float
    previous_motor_torque_nm: float = 0.0
    previous_motor_torque_rate_nm_s: float = 0.0

    def validate(self) -> None:
        """Reject impossible or unsafe sensor values before control."""
        finite_ranges = (
            ("timestamp_s", self.timestamp_s, 0.0, float("inf")),
            ("vehicle_speed_kph", self.vehicle_speed_kph, 0.0, 200.0),
            ("vehicle_acceleration_m_s2", self.vehicle_acceleration_m_s2, -20.0, 20.0),
            ("engine_speed_rpm", self.engine_speed_rpm, 0.0, 10_000.0),
            ("battery_voltage_v", self.battery_voltage_v, 0.001, 2_000.0),
            ("battery_current_a", self.battery_current_a, -10_000.0, 10_000.0),
            ("battery_temperature_c", self.battery_temperature_c, -80.0, 150.0),
            ("inverter_temperature_c", self.inverter_temperature_c, -80.0, 200.0),
            ("motor_temperature_c", self.motor_temperature_c, -80.0, 250.0),
            ("dc_bus_voltage_v", self.dc_bus_voltage_v, 0.001, 2_000.0),
            ("wheel_torque_request_nm", self.wheel_torque_request_nm, -20_000.0, 20_000.0),
            ("previous_motor_torque_nm", self.previous_motor_torque_nm, -20_000.0, 20_000.0),
            ("previous_motor_torque_rate_nm_s", self.previous_motor_torque_rate_nm_s, -100_000.0, 100_000.0),
        )
        for name, value, lower, upper in finite_ranges:
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        bounded = (
            ("engine_load_pct", self.engine_load_pct, 0.0, 100.0),
            ("accelerator_pct", self.accelerator_pct, 0.0, 100.0),
            ("brake_pct", self.brake_pct, 0.0, 100.0),
            ("battery_soc_pct", self.battery_soc_pct, 0.0, 100.0),
            ("battery_soh_pct", self.battery_soh_pct, 0.0, 100.0),
        )
        for name, value, lower, upper in bounded:
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        if not 0 <= self.vehicle_gear <= 20:
            raise ValueError("vehicle_gear must be in [0, 20]")


@dataclass(frozen=True)
class SafetyLimits:
    """Conservative placeholder limits; replace with approved component data."""

    minimum_soc_pct: float = 20.0
    maximum_soc_pct: float = 90.0
    maximum_battery_temperature_c: float = 45.0
    maximum_inverter_temperature_c: float = 85.0
    maximum_motor_temperature_c: float = 150.0
    maximum_motor_torque_nm: float = 3_000.0
    maximum_motor_torque_rate_nm_s: float = 4_000.0
    maximum_regenerative_torque_nm: float = 2_000.0


@dataclass(frozen=True)
class ObjectiveWeights:
    """Normalized weights for the research objective, not monetary values."""

    fuel: float = 1.0
    comfort: float = 0.8
    safety: float = 5.0
    battery_energy: float = 0.8
    battery_health: float = 1.2
    motor_durability: float = 1.0
    mechanical_brake_wear: float = 1.0

    def validate(self) -> None:
        values = (
            self.fuel,
            self.comfort,
            self.safety,
            self.battery_energy,
            self.battery_health,
            self.motor_durability,
            self.mechanical_brake_wear,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            raise ValueError("objective weights must be finite and non-negative")


@dataclass(frozen=True)
class ControllerCommand:
    """Bounded supervisory command sent to a lower-level torque controller."""

    motor_torque_nm: float
    regenerative_torque_nm: float
    friction_brake_request_pct: float
    fallback_active: bool
    reason: str
