---
title: 'Attitude Representation - Quaternions'
date: 2022-03-15
url: /quaternion/
description: "" 
summary: "" 
showToc: true
math: true
disableAnchoredHeadings: false
tags:
  - Attitude
  - Attitude Representation
---

[Euler Angles](/euler-angles/)

- [Euler Parameters (Quaternions) representation](#euler-parameters-quaternions-representation)
- [References:](#references)



Euler Parameters (Quaternions) representation
------

A four-element vector with three imaginary and one real component is known as Quaternion. These hypercomplex numbers are optimum for numerical stability and memory load. The Euler parameters are a four-dimensional vector that can be used to represent the orientation of a rigid body. The Euler parameters are defined as:
<div>
$$ q = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix} $$
</div>

where $q_0$ is the scalar part and $q_1$, $q_2$, and $q_3$ are the vector part. It could be written as:
<div>
$$ q = q_0 + q_1i + q_2j + q_3k $$
</div>

where $i$, $j$, and $k$ are the imaginary unit vectors and
<div>
$$ i^2 = j^2 = k^2 = ijk = -1 $$
</div>

<div>
$$ \mathbb{q} = (q_0 , \mathbf{q}_v) $$
</div>

where $ \mathbf{q}_v = (q_1 , q_2 , q_3) $ is the vector part of the quaternion. The quaternion is a unit quaternion if $ \mathbb{q} \cdot \mathbb{q}^* = 1 $, where $ \mathbb{q}^* $ is the conjugate of $ \mathbb{q} $.

It is noticeable that some authors may use left-handed quaternions witch is defined by:
<div>
$$ \mathbf{q} = iq_1 + jq_2 + kq_3 + q_0 \\ ijk = 1 $$
</div>
This representation has no fundamental implications but will change the details of formulation.

Quaternions do not have any singularity such as Euler angles. However, due to the lack of independence of components, it may present difficulties in the application of the filter equations. The quaternion is not unique, and the mirror quaternion will result in the same rotation. This is a purely mathematical representation and based upon single rotation theta around vector e with angle. It could not be used for visualization.

Quaternion also, can be used to describe the axis-angle representation by:

<div>
$$ \mathbf{q} = \begin{bmatrix}q_w \\ q_x \\ q_y \\ q_z\end{bmatrix} = \begin{bmatrix} \cos\frac{\theta}{2} \\ v_x \sin\frac{\theta}{2} \\ v_y \sin\frac{\theta}{2} \\ v_z \sin\frac{\theta}{2} \end{bmatrix} = \begin{bmatrix} \cos\frac{\theta}{2} \\ \mathbf{v} \sin\frac {\theta}{2} \end{bmatrix} $$
</div>

Also, the quaternion can be expressed in $4 \times 4$ skew-symmetric matrix form

<div>
$$ Q = \begin{bmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ q_1 & q_0 & -q_3 & q_2 \\ q_2 & q_3 & q_0 & -q_1 \\ q_3 & -q_2 & q_1 & q_0 \end{bmatrix} $$
</div>

The quaternion represents the attitude of frame $A$ relative to frame $B$ defined by the following equation: 

<div>
$$ {}^{A}_{B}\mathbf{q}={}^{B}_{A}\mathbf{q}^* $$
</div>

where $ {}^{A}_{B}\mathbf{q} $  is the quaternion that represents the attitude of frame $ A $ relative to frame $ B $.

$ {}^{B}_{A}\mathbf{q}^* $ is the conjugate of the quaternion that represents the attitude of frame $ A $ relative to frame $ B $. The $ \mathbf{q}^* $ (conjugate of the quaternion $ \mathbf{q} $) gives the inverse rotation.

The relationship between quaternions and Euler angles based on $zyx$ sequence can be calculated using the following:

<div>
$$ \mathbf{q} = \begin{bmatrix} \cos\frac{\theta_x}{2} \cos\frac{\theta_y}{2} \cos\frac{\theta_z}{2} + \sin\frac{\theta_x}{2} \sin\frac{\theta_y}{2} \sin\frac{\theta_z}{2} \\ \sin\frac{\theta_x}{2} \cos\frac{\theta_y}{2} \cos\frac{\theta_z}{2} - \cos\frac{\theta_x}{2} \sin\frac{\theta_y}{2} \sin\frac{\theta_z}{2} \\ \cos\frac{\theta_x}{2} \sin\frac{\theta_y}{2} \cos\frac{\theta_z}{2} + \sin\frac{\theta_x}{2} \cos\frac{\theta_y}{2} \sin\frac{\theta_z}{2} \\ \cos\frac{\theta_x}{2} \cos\frac{\theta_y}{2} \sin\frac{\theta_z}{2} - \sin\frac{\theta_x}{2} \sin\frac{\theta_y}{2} \cos\frac{\theta_z}{2} \end{bmatrix} $$
</div>

where $ \theta_x $, $ \theta_y $, and $ \theta_z $ are the Euler angles.

Also, the Euler angles can be calculated using the following:

<div>
$$ \phi = \arctan\left(\frac{2(q_0q_1 + q_2q_3)}{1 - 2(q_1^2 + q_2^2)}\right) $$
</div>

<div>
$$ \theta = \arcsin\left(2(q_0q_2 - q_3q_1)\right) $$
</div>

<div>
$$ \psi = \arctan\left(\frac{2(q_0q_3 + q_1q_2)}{1 - 2(q_2^2 + q_3^2)}\right) $$
</div>
Since there are 12 different Euler angles sets, there are 12 quaternion to Euler angles conversion equation.

The quaternion can be used to represent the rotation matrix $C_{\psi\theta\phi}$ as:
<div>
$$ C_{\psi\theta\phi} = \begin{bmatrix} q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1q_2 - q_0q_3) & 2(q_1q_3 + q_0q_2) \\ 2(q_1q_2 + q_0q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2q_3 - q_0q_1) \\ 2(q_1q_3 - q_0q_2) & 2(q_2q_3 + q_0q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2 \end{bmatrix} $$
</div>


[Other Attitude Representations](/attitude-representations-others/)


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