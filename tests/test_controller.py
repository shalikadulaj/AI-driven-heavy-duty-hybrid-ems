import pytest

from global_ems import ConstrainedEMSController, EMSMeasurement
from global_ems.rl import ConstrainedDoubleDQN, SyntheticEMSEnv, train_double_dqn


def measurement(**overrides: object) -> EMSMeasurement:
    values = {
        "timestamp_s": 0.0,
        "vehicle_speed_kph": 50.0,
        "vehicle_acceleration_m_s2": 0.0,
        "engine_speed_rpm": 1200.0,
        "engine_load_pct": 45.0,
        "vehicle_gear": 5,
        "accelerator_pct": 30.0,
        "brake_pct": 0.0,
        "battery_soc_pct": 60.0,
        "battery_soh_pct": 95.0,
        "battery_voltage_v": 650.0,
        "battery_current_a": 10.0,
        "battery_temperature_c": 25.0,
        "inverter_temperature_c": 45.0,
        "motor_temperature_c": 70.0,
        "dc_bus_voltage_v": 650.0,
        "wheel_torque_request_nm": 1200.0,
    }
    values.update(overrides)
    return EMSMeasurement(**values)


def test_controller_returns_bounded_propulsion_command():
    command = ConstrainedEMSController().propose(measurement(), policy_torque_nm=10000.0)
    assert 0.0 <= command.motor_torque_nm <= 3000.0
    assert not command.fallback_active


def test_braking_prefers_regeneration_before_friction_brakes():
    command = ConstrainedEMSController().propose(
        measurement(brake_pct=70.0, wheel_torque_request_nm=-1000.0),
    )
    assert command.regenerative_torque_nm > 0.0
    assert command.friction_brake_request_pct < 70.0


def test_thermal_limit_forces_safe_fallback():
    command = ConstrainedEMSController().propose(measurement(battery_temperature_c=46.0))
    assert command.fallback_active
    assert command.motor_torque_nm == 0.0
    assert command.friction_brake_request_pct == 0.0


def test_invalid_measurement_is_rejected():
    with pytest.raises(ValueError):
        ConstrainedEMSController().propose(measurement(battery_soc_pct=120.0))


def test_synthetic_environment_exposes_all_discrete_actions():
    env = SyntheticEMSEnv(horizon=3, seed=4)
    state = env.reset()
    assert state.shape == (17,)
    _, _, _, info = env.step(5)
    assert "torque_nm" in info


def test_double_dqn_trains_and_proposes_through_safety_shield(tmp_path):
    agent, returns = train_double_dqn(episodes=2, seed=8)
    assert len(returns) == 2
    env = SyntheticEMSEnv(horizon=1, seed=9)
    env.reset()
    command = agent.propose_command(env._measurement())
    assert -3000.0 <= command.motor_torque_nm <= 3000.0
    assert isinstance(agent, ConstrainedDoubleDQN)
    checkpoint = tmp_path / "policy.npz"
    agent.save(checkpoint)
    restored = ConstrainedDoubleDQN.load(checkpoint)
    observation = env.reset()
    assert restored.select_action(observation) == agent.select_action(observation)
