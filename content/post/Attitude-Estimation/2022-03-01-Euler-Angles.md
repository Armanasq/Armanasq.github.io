---
title: 'Attitude Representation - Euler Angles'
date: 2022-03-01
url: /euler-angles/
description: "" 
summary: "" 
showToc: true
math: true
disableAnchoredHeadings: false
commentable: true
tags:
  - Attitude
  - Attitude Representation
---

[Attitude Representation](/attitude-representation/)

- [Euler Angles Representation](#euler-angles-representation)
- [References:](#references)


Euler Angles Representation
------

A vector of three angles that represent the attitude of the coordinate frame $ i $ with respect to the coordinate frame $ j $ is called Euler angles. Euler angles are the most commonly used attitude representation because it's easy to use and understand. One of Euler angles' obvious advantages is their intuitive representation.
<div>
$$ \text{Euler angles} = \begin{bmatrix} \phi \\ \theta \\ \psi \end{bmatrix} $$
</div>

where $\phi$, $\theta$, and $\psi$ are the rotation angles about the $x$, $y$, and $z$ axes, respectively. The Euler angles are defined as follows:
<div>
$$  \phi = \arctan\left(\frac{R_{32}}{R_{33}}\right) \\ \theta = \arcsin\left(-R_{31}\right) \\ \psi = \arctan\left(\frac{R_{21}}{R_{11}}\right)  $$
</div>
where $R_{ij}$ is the element of the rotation matrix $R$.

* **Roll**: Rotation around the x-axis with angle $ \phi $
<div>
$$ C_{\phi} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & cos(\phi) & sin(\phi) \\ 0 & -sin(\phi) & cos(\phi) \end{bmatrix} $$
</div>
* **Pitch**: Rotation around the y-axis with angle $ \theta $
<div>
$$ C_{\theta} = \begin{bmatrix} cos(\theta) & 0 & -sin(\theta) \\ 0 & 1 & 0 \\ sin(\theta) & 0 & cos(\theta) \end{bmatrix} $$
</div>
* **Yaw**: Rotation around the z-axis with angle $ \psi $
<div>
$$ C_{\psi} = \begin{bmatrix} cos(\psi) & sin(\psi) & 0 \\ -sin(\psi) & cos(\psi) & 0 \\ 0 & 0 & 1 \end{bmatrix} $$
</div>
Euler angles represent three consecutive rotations, and they could be defined in twelve different orders. The most common order is the yaw-pitch-roll (YPR) order, which is also called the z-y-x order. The rotation matrix can be written as:
<div>
$$ C_{\psi\theta\phi} = C_{\psi}C_{\theta}C_{\phi} $$
</div>
<div>
$$ C_{\psi\theta\phi} = \begin{bmatrix} cos(psi)cos(\theta) & cos(psi)sin(\theta)sin(\phi)-sin(psi)cos(\phi) & cos(psi)sin(\theta)cos(\phi)+sin(psi)sin(\phi) \\ sin(psi)cos(\theta) & sin(psi)sin(\theta)sin(\phi)+cos(psi)cos(\phi) & sin(psi)sin(\theta)cos(\phi)-cos(psi)sin(\phi) \\ -sin(\theta) & cos(\theta)sin(\phi) & cos(\theta)cos(\phi) \end{bmatrix} $$
</div>
The Euler angles of the rotation matrix $C_{\phi\theta\psi}$ can be written as:
<div>
$$ \phi = \arctan\left(\frac{C_{32}}{C_{33}}\right) $$
</div><div>
$$ \theta = \arctan\left(\frac{C_{32}}{\sqrt{1-C_{32}^2}}\right) $$
</div><div>
$$ \psi = \arctan\left(\frac{C_{31}}{C_{33}}\right) $$
<div>
The Euler angles are not unique. For example, the Euler angles $ (0,0,0) $ and $ (2\pi,2\pi,2\pi) $ represent the same rotation. The Euler angles are also not invariant to the order of the rotations. For example, the Euler angles $ R_{x,y,z}(0,0,0) $ and $ R_{z,y,x}(0,0,0) $ represent the same rotation, but the rotation matrix is different.


Three rotation angles $\phi$, $\theta$, and $\psi$ are about the sequential displaced body-fixed axes, and twelve different sequences are possible that can be used for the same rotation. The location of each sequential rotation depends on the preceding rotation, and there are divided into two main categories:
<ol> 
  <li> <b>Symmetric sequences</b>: The first and third rotations are performed around the same axis, second rotation is performed around one of the two others:</li>
$$ R_{i,j,i}(\alpha, \beta, \gamma) = R_i(\alpha)R_j(\beta)R_i(\gamma) $$
<center> Symmetric sequence $ (i,j,i)$, $ i \ne j$, $ \alpha, \beta, \gamma \in \mathbb{R}$ </center>

  <li> <b>Asymmetric sequences</b>: All rotations performed around three different axes: </li>
$$ R_{i,j,k}(\alpha, \beta, \gamma) = R_i(\alpha)R_j(\beta)R_k(\gamma) $$
<center> Asymmetric sequence $ (i,j,k)$, $ i \ne j \ne k \ne i$, $ \alpha, \beta, \gamma \in \mathbb{R}$ </center>

</ol>

These angles are not unique, and the mirror angles will result in the same rotations.

<ol>
  <li> For <b><i>Symmetric sequences</i></b>: $ R(\alpha, \beta, \gamma) = R(\alpha + \pi, -\beta,\gamma - \pi) $ </li>
  <li> For <b><i>Asymmetric sequences</i></b>: $ R(\alpha, \beta, \gamma) = R(\alpha + \pi, \pi -\beta,\gamma - \pi) $ </li>
</ol>

The main disadvantages of Euler angles are:
<ol>
<li>Singularity
</li>

<li>Non-uniqueness </li>

<li>Non-invariance </li>

<li> Less accuracy for integration of attitude incremental changes over time</li>
</ol>

At $ \theta = \left(\pm\frac{\pi}{2}\right) $ the singularities will occur and usually known as mathematical gimble lock where to axes are parallel to each other.

[Quaternion](/quaternion/)

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