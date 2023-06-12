---
title: 'Attitude Representation - Other'
date: 2022-04-01
url: /attitude-representations-others/
description: "" 
summary: "" 
showToc: true
math: true
disableAnchoredHeadings: false
tags:
  - Attitude
  - Attitude Representation
---

[Quaternion](/quaternion/)

- [Gibbs Vector / Rodrigues Parameter Representation](#gibbs-vector--rodrigues-parameter-representation)
- [Modified Rodrigues Parameters](#modified-rodrigues-parameters)
- [Cayley-Klein](#cayley-klein)
- [References:](#references)


Gibbs Vector / Rodrigues Parameter Representation
------
The Gibbs vector also known as Rodrigues Parameter is a set of three parameters denoted by $ g $ (or $ P $) and can be directly derived from axis-angle $ (e, \theta) $ or quaternion representation as follows:

<div>
$$ \mathbf{g} = \frac{\mathbf{q}_{v}}{q_0} $$
</div>

<div>
$$ \mathbf{g} = \frac{e \sin\frac{\theta}{2}}{\cos\frac{\theta}{2}} $$
</div>

where $ \mathbf{q}_{v} $ is the vector part of the quaternion, $ e $ is the unit vector of the axis of rotation, and $ \theta $ is the angle of rotation.

The Gibbs vector is a unit vector that represents the axis of rotation and the magnitude of the vector represents the angle of rotation. The Gibbs vector can be used to represent the rotation matrix $C_{\psi\theta\phi}$ as:

<div>
$$ C_{\psi\theta\phi} = \begin{bmatrix} 1 - 2(g_2^2 + g_3^2) & 2(g_1g_2 - g_3) & 2(g_1g_3 + g_2) \\ 2(g_1g_2 + g_3) & 1 - 2(g_1^2 + g_3^2) & 2(g_2g_3 - g_1) \\ 2(g_1g_3 - g_2) & 2(g_2g_3 + g_1) & 1 - 2(g_1^2 + g_2^2) \end{bmatrix} $$
</div>

where $ g_1 $, $ g_2 $, and $ g_3 $ are the components of the Gibbs vector.

The Gibbs vector components expressed in DCM can be calculated using the following:

<div>
$$ g_1 = \frac{R_{23}-R_{32}}{1+R_{11}+R_{22}+R_{33}} $$
</div>

<div>
$$ g_2 = \frac{R_{31}-R_{13}}{1+R_{11}+R_{22}+R_{33}} $$
</div><div>
$$ g_3 = \frac{R_{12}-R_{21}}{1+R_{11}+R_{22}+R_{33}} $$
</div>
where $ R_{ij} $ is the element of the rotation matrix.

The Rodrigues Parameter preferred as an attitude error representation because it is a unit vector and it is easy to calculate the attitude error between two quaternions. The attitude error between two quaternions can be calculated using the following:
<div>
$$ \mathbf{g}_{error} = \frac{2\mathbf{q}_{v}}{q_0} $$
</div>
The Gibbs vector components expereince a singularity at $ \theta = \pi $, which is the same as the Euler angles. The Gibbs vector is not a good representation for small rotations.

Modified Rodrigues Parameters
------

In attitude filter design Modified Rodrigues Parameters (MRP) is preferred for attitude error representation. The MRP is a set of three parameters denoted by $ \mathbf{m} $ and can be directly derived from axis-angle $ (e, \theta) $ or quaternion representation as follows: 
<div>
$$ \mathbf{m} = \frac{\mathbf{q}_{v}}{1+q_0} $$
</div><div>
$$ \mathbf{m} = \frac{e \sin\frac{\theta}{2}}{1+\cos\frac{\theta}{2}} $$
</div>
where $ \mathbf{q}_{v} $ is the vector part of the quaternion, $ e $ is the unit vector of the axis of rotation, and $ \theta $ is the angle of rotation.

Due to above equation the maximum equivalent rotation to describe is $ \pm 360^{\circ}$ (the singularity occurs in $ \pm 360^{\circ}$).

Cayley-Klein
------

The Cayley-Klein parameters are consisting of 4 parameters which are closely related to the quaternions and denoted by matrix $ \mathbf{K}_{2\times 2} $.
<div>
$$ K = \begin{bmatrix} \alpha & \beta \\ \gamma & \sigma \end{bmatrix} $$
</div>
and satisfy the constraints
<div>
$$ \alpha \bar{\alpha} + \gamma \bar{\gamma} = 1 \\ \alpha \bar{\alpha} + \beta \bar{\beta} = 1 \\ \alpha \bar{\beta} + \gamma \bar{\sigma} = 0 \\ \alpha \sigma + \beta \gamma = 1 \\ \beta = -\bar{\gamma} \\ \sigma = \bar{\alpha} $$
</div>
where $ \alpha $, $ \beta $, $ \gamma $, and $ \sigma $ are the Cayley-Klein parameters and $ \bar{\alpha} $, $ \bar{\beta} $, $ \bar{\gamma} $, and $ \bar{\sigma} $ are the conjugate of the Cayley-Klein parameters.

The corresponding quaternions are defined_as:
<div>
$$ \mathbf{q}_K = \begin{bmatrix} \frac{ \alpha + \sigma }{2} \\ \frac{ -i(\beta + \gamma) }{2} \\ \frac{ \beta - \gamma }{2} \\ \frac{ -i(\alpha - \sigma) }{2} \end{bmatrix} $$
</div>

References:
------
[1] Markley, F. Landis, and John L. Crassidis. Fundamentals of spacecraft attitude determination and control. Vol. 1286. New York, NY, USA:: Springer New York, 2014. <br>
[2] Junkins, John L., and Hanspeter Schaub. Analytical mechanics of space systems. American Institute of Aeronautics and Astronautics, 2009. <br>
[3] De Ruiter, Anton H., Christopher Damaren, and James R. Forbes. Spacecraft dynamics and control: an introduction. John Wiley & Sons, 2012. <br>
[4] Wertz, James R., ed. Spacecraft attitude determination and control. Vol. 73. Springer Science & Business Media, 2012. <br>
[5] Vepa, Ranjan. Dynamics and Control of Autonomous Space Vehicles and Robotics. Cambridge University Press, 2019. <br>
[6] Shuster, Malcolm D. "A survey of attitude representations." Navigation 8.9 (1993): 439-517. <br>
[7] Markley, F. Landis. "Attitude error representations for Kalman filtering." Journal of guidance, control, and dynamics 26.2 (2003): 311-317. <br>
[8] Markley, F. Landis, and Frank H. Bauer. Attitude representations for Kalman filtering. No. AAS-01-309. 2001. <br>