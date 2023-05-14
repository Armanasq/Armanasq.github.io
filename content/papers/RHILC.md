---
title: "A Receding Horizon Iterative Learning (RHILC) Approach to Formation Control of Multi-Agent Non-Holonomic System"
collection: publications
permalink: /publication/RHILC
excerpt: 'A multi-agent formation control strategy using a receding horizon controller that utilizes a nonlinear cost function solved via the steepest descent interactive method.'
date: 2022-9-21
venue: 'Aeronautical Journal - Under Review'
#paperurl: 'http://academicpages.github.io/files/paper1.pdf'
#citation: ''
---
M. H. Sabour, A. Asgharpoor.

This paper proposes a multi-agent formation control strategy using a receding horizon controller that utilizes a nonlinear cost function solved via the steepest descent interactive method. The objective is to reach the desired position while avoiding inter-agent collision and divergence of agents. The proposed controller is used to control non-holonomic kinematic agents in a behavioral structure. The control architecture combines the concept of prediction horizons defined in receding horizon control (model predictive control) with iterative learning controllers (ILC). At each time step, the iterative learning controller is run for the period of the prediction horizon, and a control sequence is obtained. The first element of this sequence is implemented in the system. Henceforth, this control technique will be denoted as Receding Horizon Iterative Learning Control (RHILC). Simulation results are presented to demonstrate the capabilities of this controller further. The results are discussed, and future work possibilities are given.


In this article, the receding horizon iterative learning control (RHILC) is considered a solution for the multi-agent non-holonomic system formation problem. To determine the dynamics of an agent, we use first-order differential equations with two inputs and three states. RHC has raised challenges regarding feasibility, whereas some remarkable work has been done, proving that system dynamics bind the stability of formation controllers. Despite technological advancements, computation processing capability is one of the severe challenges in using the RHC method. ILC is a feasible and model-independent control technique. However, it is bound to batch or periodic processes. The ILC algorithm is unsuitable for the formation control problem because this controller must be calculated offline and could not correct the perturbation and environmental disturbances. The results indicate that combining these two techniques leads to acceptable performance levels and a high degree of stability. Moreover, the computational load was reduced because an ILC problem was solved instead of an optimization problem in the receding horizon controller. 
A novel control algorithm is presented by combining the notion of a receding horizon with an ILC controller. This technique expands the scope of ILC controllers to tracking problems and real-time non-periodic processes. However, further studies are required regarding the stability of this controller and improving the tracking performance of moving targets. Formation control of a group of agents was carried out utilizing the proposed method. Simulation results were presented using the behavioral structure for a group of three agents. The results displayed good performance for the fixed goal problem. As expected, the tracking problem had poor performance due to the absence of the error derivative in the update formula. Future works will focus on stability criteria and tracking the performance of this control algorithm.

