# Globally Optimized AI-Driven EMS for a P4 Truck

Research scaffold for a **two-axle P4 hybrid truck** in which one axle is
driven by a diesel engine and the other axle is driven by an electric motor.
The electric axle is the rear axle according to the current installation
requirement. This repository targets a balanced objective:

1. fuel economy,
2. driver comfort,
3. functional and operational safety,
4. battery energy and health,
5. motor/inverter and mechanical durability.

The project is designed for an instrumented truck, Python plant modeling,
MATLAB/Simulink, autocode, and HIL. It is **not** a claim that the current
controller is safe for direct actuation. Real-truck deployment must proceed
through simulation, SIL, HIL, shadow mode, bounded torque supervision,
independent safety validation, and only then a separately approved actuation
path.

## What the controller must optimize

The energy-management system receives synchronized measurements from the
vehicle network and proposes rear e-axle motor torque. The objective includes:

- engine fuel and operating-point efficiency,
- battery charge/discharge energy and charge-sustaining operation,
- SoC, SoH, temperature, current, voltage, and power limits,
- battery cycle and calendar-aging indicators,
- motor and inverter temperature,
- motor torque magnitude, torque rate, thermal exposure, and switching stress,
- regenerative braking before friction braking when safe and physically
  feasible,
- driver accelerator/brake intent, selected gear, speed tracking, jerk, and
  torque smoothness,
- mechanical-brake energy and wear proxy,
- traction, wheel-torque, DC-bus, and fault constraints.

The future learned policy should not directly command hardware. It should
propose an action to a deterministic constraint/safety shield and a
lower-level certified torque controller.

## Current implementation

`src/global_ems/` contains a transparent baseline:

- typed CAN/J1939/DAQ measurement schema,
- explicit safety limits and multi-objective weights,
- bounded candidate torque search,
- regenerative/friction-brake blending,
- thermal/SoC fallback,
- policy-action clamping before actuator-facing output.

This is deliberately a safe research interface, not deep reinforcement
learning yet. A suitable research path is:

1. identify and validate the truck plant and sensors;
2. train a constrained SAC or TD3 policy in randomized simulation;
3. use domain randomization and an ensemble of plant models;
4. apply a deterministic safety shield and action-rate limiter;
5. distill or constrain the policy for embedded real-time execution;
6. compare against dynamic programming, ECMS, rule control, MPC, and the
   transparent baseline;
7. validate in SIL, HIL, shadow mode, and controlled test-track trials.

Deep RL is useful for nonlinear, preview-dependent control, but it does not
replace safety engineering, system identification, calibration, or
certification.

## Measurements to acquire

At a common timestamp, log at minimum:

- vehicle speed, acceleration, wheel speeds, selected gear, accelerator and
  brake pedals;
- engine speed, torque/load request, fuel rate if available, coolant and oil
  temperatures;
- battery pack voltage/current, SoC, BMS SoH, cell extrema, temperatures,
  contactor/fault state, charge/discharge limits;
- inverter DC voltage/current, phase current if available, inverter temperature;
- motor speed, torque estimate/request, winding and housing temperatures;
- rear e-axle speed/torque, differential and lubrication temperatures when
  available;
- friction-brake pressure/temperature and ABS/ESC interventions;
- road grade, mass/load estimate, ambient temperature, and route preview.

Signal names, scaling, timestamps, quality flags, sampling rates, and
dropout behavior must be documented from the actual truck DBC/J1939 database.
Do not infer unavailable signals from similarly named fields.

## Run locally

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

The current tests exercise measurement validation, bounded torque, regenerative
braking preference, and thermal fallback. Simulation results must be generated
only after the plant parameters and drive-cycle provenance are recorded in
`artifacts/`.

## Real-truck deployment gate

The controller must not be connected directly to actuators from this repository.
Follow the staged process in
[docs/REAL_TRUCK_DEPLOYMENT.md](docs/REAL_TRUCK_DEPLOYMENT.md). A production
system requires an independent safety concept, failure-mode analysis,
cybersecurity controls, watchdogs, redundant plausibility checks, fallback
control, driver override, logging, calibration, and approval under the
applicable vehicle and functional-safety processes.

## Scientific basis

The following peer-reviewed papers motivate health-aware and learning-based
energy management. They do **not** provide truck-specific parameters for this
repository:

1. Wang, X., Ma, S., and Wang, J. (2018). “Predictive energy management for
   hybrid electric vehicles considering extension of the battery life.”
   *Proceedings of the Institution of Mechanical Engineers, Part D*, 232(4),
   499–510. DOI:
   [10.1177/0954407017703229](https://doi.org/10.1177/0954407017703229).
2. Wu, J. et al. (2021). “Battery Thermal- and Health-Constrained Energy
   Management for Hybrid Electric Bus Based on Soft Actor-Critic DRL
   Algorithm.” *IEEE Transactions on Industrial Informatics*, 17(6), 3751–3761.
   DOI: [10.1109/TII.2020.3014599](https://doi.org/10.1109/TII.2020.3014599).
3. Schmalstieg, J., Käbitz, S., Ecker, M., and Sauer, D. U. (2014). “A holistic
   aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries.”
   *Journal of Power Sources*, 257, 325–334. DOI:
   [10.1016/j.jpowsour.2014.02.012](https://doi.org/10.1016/j.jpowsour.2014.02.012).
4. van Hasselt, H., Guez, A., and Silver, D. (2016). “Deep Reinforcement
   Learning with Double Q-learning.” *Proceedings of the AAAI Conference on
   Artificial Intelligence*, 30(1). DOI:
   [10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295).

## Scope and limitations

The rear electric axle is user-specified. Diesel-axle location, component
ratings, battery chemistry, aging coefficients, thermal limits, and CAN signal
availability require confirmation from vehicle drawings and measured data.
No fuel, life, safety, or durability result should be generalized to a real
truck until the model is calibrated and independently validated.
