# Real-truck deployment plan

## Non-negotiable safety boundary

The research controller must not directly actuate engine, inverter, motor, or
brake hardware until the vehicle manufacturer/integrator has approved the
interface and the complete safety case. A public repository cannot establish
functional safety, cybersecurity, or roadworthiness.

The first truck installation should be **read-only logging**. The second stage
should be **shadow mode**, in which the controller computes commands but a
validated production EMS controls the vehicle. Only after independent review
should a bounded torque-supervisor experiment be considered on a controlled
test track. Direct full EMS control is a separate engineering program.

## Staged workflow

### 1. Vehicle and signal identification

- Obtain the approved axle topology, torque directions, axle ratios, ratings,
  BMS limits, inverter limits, motor map, engine map, brake map, and DBC/J1939
  database.
- Define signal ownership and freshness requirements.
- Confirm whether the electric axle can provide negative torque at each speed
  and whether friction-brake blending is coordinated by the production brake
  controller.
- Measure sensor latency, quantization, dropouts, plausibility ranges, and
  timestamp synchronization.

### 2. Model calibration

Calibrate mass/load, road load, tire radius, driveline efficiency, engine fuel
map, motor/inverter efficiency, battery OCV/SoC relationship, current limits,
thermal dynamics, and brake blending using independent tests. Use separate
calibration and validation routes. Do not fit and report performance on the
same route.

Battery life requires cell/pack data over relevant temperature, SoC, C-rate,
depth-of-discharge, and rest conditions. A simple proxy is not a warranty-life
model.

### 3. SIL and HIL

- Replay time-synchronized truck logs through the controller.
- Test sensor faults, stale messages, sign errors, range errors, lost
  communications, over-temperature, over-current, low/high SoC, and
  conflicting pedal/brake inputs.
- Verify worst-case execution time, memory, numerical bounds, and watchdog
  behavior on the target ECU.
- Compare the learned controller with dynamic programming, ECMS, MPC, and a
  production/rule baseline.

### 4. Shadow mode

Run the controller on real missions without sending commands. Compare proposed
torque with the production EMS and inspect:

- safety-shield activations,
- tracking and comfort,
- battery current/temperature/SoC limits,
- motor/inverter temperatures,
- regenerative energy and friction-brake use,
- communication quality and timing.

### 5. Bounded test-track supervision

If approved by the vehicle integrator, allow only a bounded torque request
behind an independent lower-level controller. Keep hard limits outside the
learned policy. Require a physical driver override, a production fallback,
watchdogs, redundant plausibility checks, and immediate abort criteria.

### 6. Production readiness

Complete the applicable hazard analysis, FMEA/FMEDA, cybersecurity threat
analysis, software quality process, calibration management, traceability,
verification report, driver/maintenance procedures, and regulatory approval.

## Implemented and recommended deep-RL architecture

The repository includes a dependency-light NumPy Double-DQN prototype. It is
intended to verify the training interface and safety-shield integration. It
does not represent a calibrated truck, and its synthetic training results
must not be used as evidence of real-vehicle performance.

For the next plant-backed stage, use a continuous-action constrained SAC or TD3 actor for a proposed motor
torque fraction, with observations normalized from measured signals and route
preview. Train with domain randomization over mass, grade, road load,
component efficiency, battery temperature, sensor noise, latency, and
parameter uncertainty. Use an ensemble or uncertainty estimator to penalize
out-of-distribution states.

The runtime stack should be:

1. signal validation and freshness monitor;
2. deterministic safety envelope;
3. policy proposal;
4. action-rate and torque/thermal/SoC shield;
5. regenerative/friction-brake coordinator;
6. lower-level motor torque controller;
7. watchdog, fallback, driver override, and immutable event log.

The policy objective should be evaluated as a Pareto set or constrained
optimization, not one arbitrary reward number. Report fuel-equivalent use,
SoC balance, battery throughput, SoH proxy, thermal exposure, motor torque
cycling, friction-brake energy, jerk, tracking error, safety interventions,
and uncertainty by route and load condition.
