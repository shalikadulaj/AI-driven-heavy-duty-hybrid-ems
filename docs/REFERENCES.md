# References and evidence map

This list separates peer-reviewed research from engineering standards. The
papers motivate algorithm and aging choices; they do **not** supply calibrated
parameters for the truck in this repository. DOI links point to publisher or
publisher-indexed records where available.

## Peer-reviewed research

### Battery-aware hybrid-vehicle energy management

1. **Wang, X., Ma, S., & Wang, J. (2018).** Predictive energy management for
   hybrid electric vehicles considering extension of the battery life.
   *Proceedings of the Institution of Mechanical Engineers, Part D: Journal of
   Automobile Engineering, 232*(4), 499–510.
   [DOI: 10.1177/0954407017703229](https://doi.org/10.1177/0954407017703229)

   Relevance: predictive HEV energy management with a battery-fading index.
   It supports including battery-aging cost, but does not identify this
   truck's battery coefficients.

2. **Wu, J., Wei, Z., Li, W., Wang, Y., Li, Y., & Sauer, D. U. (2021).**
   Battery thermal- and health-constrained energy management for hybrid
   electric bus based on soft actor-critic DRL algorithm. *IEEE Transactions
   on Industrial Informatics, 17*(6), 3751–3761.
   [DOI: 10.1109/TII.2020.3014599](https://doi.org/10.1109/TII.2020.3014599)

   Relevance: thermal- and health-constrained deep RL for a hybrid bus. The
   bus model, cell data, and controller are not interchangeable with this
   truck without calibration.

3. **Schmalstieg, J., Käbitz, S., Ecker, M., & Sauer, D. U. (2014).** A
   holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries.
   *Journal of Power Sources, 257*, 325–334.
   [DOI: 10.1016/j.jpowsour.2014.02.012](https://doi.org/10.1016/j.jpowsour.2014.02.012)

   Relevance: chemistry- and cell-specific capacity/resistance aging model.
   It is evidence for separating calendar and cycling effects, not a valid
   parameter source for an unknown truck pack.

4. **Du, R., Hu, X., Xie, S., Hu, L., Zhang, Z., & Lin, X. (2020).** Battery
   aging- and temperature-aware predictive energy management for hybrid
   electric vehicles. *Journal of Power Sources, 473*, 228568.
   [DOI: 10.1016/j.jpowsour.2020.228568](https://doi.org/10.1016/j.jpowsour.2020.228568)

   Relevance: couples battery aging and thermal behavior to predictive energy
   management. The reported model must not be transplanted to this pack
   without chemistry and thermal calibration.

5. **Huang, R., He, H., Zhao, X., Wang, Y., & Li, M. (2022).** Battery
   health-aware and naturalistic data-driven energy management for hybrid
   electric bus based on TD3 deep reinforcement learning algorithm.
   *Applied Energy, 321*, 119353.
   [DOI: 10.1016/j.apenergy.2022.119353](https://doi.org/10.1016/j.apenergy.2022.119353)

   Relevance: health-aware TD3 and naturalistic driving data for a hybrid bus;
   useful precedent for the future continuous-action truck policy.

6. **Iqbal, N., Wang, H., Zheng, Z., & Yao, M. (2024).** Reinforcement
   learning-based heuristic planning for optimized energy management in
   power-split hybrid electric heavy duty vehicles. *Energy, 297*, 131773.
   [DOI: 10.1016/j.energy.2024.131773](https://doi.org/10.1016/j.energy.2024.131773)

   Relevance: directly addresses RL-based EMS planning for heavy-duty hybrid
   vehicles and varying slope conditions.

### Deep reinforcement learning

7. **Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).** Human-level
   control through deep reinforcement learning. *Nature, 518*, 529–533.
   [DOI: 10.1038/nature14236](https://doi.org/10.1038/nature14236)

   Relevance: DQN with experience replay and a target network.

8. **van Hasselt, H., Guez, A., & Silver, D. (2016).** Deep reinforcement
   learning with Double Q-learning. *Proceedings of the AAAI Conference on
   Artificial Intelligence, 30*(1), 2094–2100.
   [DOI: 10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295)

   Relevance: separates action selection and value evaluation to reduce
   overestimation; the current prototype implements this Double-DQN idea.

9. **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).** Soft
   Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
   Stochastic Actor. *Proceedings of the 35th International Conference on
   Machine Learning*, PMLR 80, 1861–1870.
   [PMLR record](https://proceedings.mlr.press/v80/haarnoja18b.html)

   Relevance: candidate continuous-action algorithm for a future calibrated
   torque policy.

10. **Fujimoto, S., van Hoof, H., & Meger, D. (2018).** Addressing function
   approximation error in actor-critic methods. *Proceedings of the 35th
   International Conference on Machine Learning*, PMLR 80, 1587–1596.
   [PMLR record](https://proceedings.mlr.press/v80/fujimoto18a.html)

   Relevance: TD3-style delayed policy updates and clipped double critics for
   continuous control.

## Engineering standards and interfaces

These are authoritative engineering references, not peer-reviewed research
papers:

11. **ISO 26262:2018, Road vehicles — Functional safety.**
   [ISO publication page](https://www.iso.org/publication/PUB200262.html)

   Relevance: safety lifecycle and safety-related electrical/electronic
   systems. Compliance cannot be inferred from this research repository.

12. **SAE J1939_202603, Serial Control and Communications Heavy-Duty Vehicle
   Network — Top-Level Document.**
   [SAE record](https://doi.org/10.4271/J1939_202603)

   Relevance: heavy-duty vehicle network terminology and message structure.
   The actual truck DBC and OEM signal definitions remain project inputs.

## Evidence-to-implementation map

| Repository element | Evidence or standard | What remains to be measured |
|---|---|---|
| Replay buffer and target network | Mnih et al. (2015) | Generalization to truck missions |
| Double-DQN update | van Hasselt et al. (2016) | Matched comparison against DQN, MPC, ECMS |
| Future continuous torque policy | Haarnoja et al. (2018); Fujimoto et al. (2018) | Stability under truck actuator limits |
| Health-aware reward | Wang et al. (2018); Wu et al. (2021) | Pack-specific aging and thermal calibration |
| Cycle/calendar aging separation | Schmalstieg et al. (2014) | Cell chemistry, pack configuration, cooling |
| Deployment safety process | ISO 26262 | HARA, FMEA/FMEDA, verification, approval |
| Signal transport | SAE J1939 | OEM DBC, scaling, freshness, fault behavior |

## Citation policy

No reference above is used to claim that the current synthetic environment
predicts real fuel consumption, battery life, motor durability, or safety. Any
future performance claim must identify the plant version, calibration data,
drive cycles, random seeds, baselines, confidence intervals, and validation
status.

