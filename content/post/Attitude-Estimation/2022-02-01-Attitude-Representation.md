---
title: 'Attitude Representation'
date: 2022-01-17
url: /attitude-representation/
author: "Arman Asgharpoor Golroudbari"
description: "" 
summary: "" 
showToc: true
disableAnchoredHeadings: false
math: true
commentable: true
tags:
  - Attitude
  - Attitude Representation
---
- [Direction Cosine Matrix (DCM)](#direction-cosine-matrix-dcm)
- [Axis-Angle Representation](#axis-angle-representation)
- [References:](#references)


[Attitude](/attitude/)

Attitude representation is a set of coordinates that fully describe a rigid body’s orientation with respect to a reference frame. There are an infinite number of attitude representations, each of which has strengths and weaknesses. Choosing the proper attitude representation depends on the estimation algorithm, type of the moving object (e.g. satellite, spacecraft), type of mission, and reference frame selection. Attitude representation impacts mathematical complexity, geometrical singularities, and operational range, so it's crucial to choose the proper representation for the objectives. At least, three coordinates are needed to describe the attitude in a 3D space that has at least one singularity. Singularities can be avoided by using four or more coordinates, but even the use of four coordinates does not guarantee their avoidance.

There are various attitude representations that are common in the industry, such as Direction Cosine Matrix, Euler angles, Euler Parameters (Quaternions), Gibb's vectors, and so on. We will describe a few of them below.

To maintain consistency in mathematical notations, two reference frames (as a reference frame) and (as a body frame) have been defined as follows:
<div>
$$ 
N \equiv  \begin{bmatrix} n_1 \\ n_2 \\ n_3 \end{bmatrix},   B \equiv  \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}  
$$
</div>


Direction Cosine Matrix (DCM)
------

In mathematics, a direction cosine matrix (DCM) is a matrix that transforms coordinate reference frames. Attitude Matrix, also known as DCM, is the most fundamental and redundant method of describing relative attitudes.
<div>
$$
\mathbf{R} =
\begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{bmatrix} \in \mathbb{R}^{3\times 3}
$$
</div>

There are nine parameters, of which six are redundant due to orthogonality. The DCM elements can be described as the dot product of coordinate system axes, which express the base vector as follows:
<div>
$$ DCM = \begin{bmatrix} b_1 \cdot n_1 & b_1 \cdot n_2 & b_1 \cdot n_3 \\ b_2 \cdot n_1 & b_2 \cdot n_2 & b_2 \cdot n_3 \\ b_3 \cdot n_1 & b_3 \cdot n_2 & b_3 \cdot n_3\end{bmatrix} $$
</div>
In the other hand, the cosine of three angles between each body vector $ b_i, (i=1,2,3) $ and three axes $ n_i, (i=1,2,3) $ are called the direction cosine matrix.
<div>
$$ b_i = cos(\alpha_{i1}\mathbf{n}_1) + cos(\alpha_{i2}\mathbf{n}_2) + cos(\alpha_{i3}\mathbf{n}_3) \\ i=1,2,3 $$
</div>

So, the direction cosine matrix can be rewritten by:

<div>
$$ DCM = \begin{bmatrix} cos(\alpha_{11}) & cos(\alpha_{12}) & cos(\alpha_{13}) \\ cos(\alpha_{21}) & cos(\alpha_{22}) & cos(\alpha_{23}) \\ cos(\alpha_{31}) & cos(\alpha_{32}) & cos(\alpha_{33})  \end{bmatrix} $$
</div>


So,
<div>
$$ \hat{\mathbf{b}} = \text{DCM} \hat{\mathbf{n}} $$
</div>


where $\hat{\mathbf{b}}$ and $\hat{\mathbf{n}}$ are the unit vectors of the body and reference frames, respectively.

Axis-Angle Representation
------

Euler’s theorem states that all rotations of a solid object can be expressed as single rotation $ \theta $ about a unit length axis $ e $ in the rotation plane. In other words, each orthogonal matrix $ R $ has a specified unit vector rotation axis donated $ e $, known as Euler axis, and a single rotation angle $ \theta $ is called Euler angle. The axis angle representation can be written as:

$$ \theta \mathbf{e}= \begin{bmatrix} \theta e_1 \\ \theta e_2 \\ \theta e_3 \end{bmatrix} $$

where

$$ \mathbf{e} = \begin{bmatrix} e_1 \\ e_2 \\ e_3 \end{bmatrix} $$

and

$$ \|\mathbf{e}\| = 1 $$

Since, $(e,\theta)$ and $(-e,-\theta)$ correspond to the same rotation, it’s not a unique representation. The axis-angle representation is not a good choice for attitude estimation because it has a singularity at $\theta = \pi$. The axis-angle representation is also not a good choice for attitude control because it is not a linear representation. The axis-angle representation is a good choice for attitude visualization. The axis-angle representation is also a good choice for attitude initialization.

[Euler Angles](/euler-angles/)


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