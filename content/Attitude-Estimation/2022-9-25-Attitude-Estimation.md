---
title: "Attitude Estimation"
date: 2022-09-25
url: /attitude-estimation/attitude-estimation/
description: "" 
summary: "" 
showToc: true
math: true
disableAnchoredHeadings: false
tags:
  - Navigation
  - Attitude Estimation
  - Orientaion Estimation
---

## Abstract


## Introduction

Achieving accurate and precise attitude determination or estimation is needed to perform successful navigation. Each flying vehicle either in air or space, needs to determine and control its attitude based on mission requirements. Vast variety of instruments/sensors and algorithm have been developed in the last decades; they are distinct by their cost and complexity. Use an accurate sensor will exponentially increase the cost which could exceed the budget. A solution for increase the accuracy with low cost is to use multi sensors (homogenous or heterogenous); multiple sensors could sense a quantity from different perspective or sense multi quantities to reduce the error and uncertainty. Multiple sensors fuse their data to achieve more accurate quantity, this method usually called as Multi-Data Sensor Fusion (MSDF). MSDF use mathematical methods to reduce noise, uncertainty and also estimate the quantity based on priori data and it could be utlized for attitude determiation. Attitude determination methods could be broadly divided in two classes, single-point and recursive estimation. First method calculates the attitude by use of two or more vector measurements at a single point of time. Instead, recursive methods use the combination of measurements over time and the system mathematical model. A precise attitude determination is dependent on sensor’s precision, accurate system modeling, and the information processing method. Obtaining this precision is considered a challenging navigation problem due to system modeling, process, and measurements errors. Increase the sensor’s precision may exponentially increase the cost; sometimes, achieving the precision requirements will only be possible for an exorbitant cost.

One approach for determining the attitude is using inertial navigation algorithms based inertial sensors. Inertial Navigation is based on the Dead Reckoning method. In this method, different types of inertial sensors are used such as accelerometer and gyroscope which called Inertial Measurement Unit (IMU). A moving object's position, velocity, and attitude can be determined using numerical integration of IMU measurements.

Using low-cost Micro Electro Mechanical Systems (MEMS) based Inertial Measurement Unit (IMU) has been grown in the past decade. Due to recent advances in MEMS technology, IMUs became smaller, cheaper, and more accurate, and they are now available for use in mobile robots, smartphones, drones, and autonomous vehicles. This sensors suffers from noise and bias, which affect dirctly the performance attitude estimation alogrithm.
In the past decades, different MSDF techniques and Deep Learning models have been developed to tackle this problem and increase the accuracy and reliability of attitude estimation techniques.

Attitude can be represented in many different forms. The Tait-Bryan angles (also called Euler angles) are the most familiar form and known as yaw, pitch, and roll (or heading, elevation, and bank). Engineers widely use rotation matrix and quaternions, but the quaternions are less intuitive.



## Related works

In the past decade, much research has been conducted on the inertial navigation techniques. These studies could roughly divided in three categories, estimation methods, Multi-Data Sensor Fusion (MSDF) techinques, and evolutionary/AI algorithms. Kalman Filter family (i.e., EKF, UKF, MEKF) and other commonly used algorithms such as Madgwick, and Mahony are based on the dynamic model of the system. Kalman filter first introduced in [], and its vairents such as EKF, UKF, and MEKF have been implemented for attitude estimation applications.

In [] Carsuo et al. compared different sensor fusion algorithms for inertial attitude estimation. this comparative study showed that Sensor Fusion Algorithms (SFA) performance are highly depended to parameters tuning and fixed parameter values are not suitable for all applications. So, the parameter tuning is one the disadvantages of conventioal attitude estimation method. This problem could be tackeld by using evolutionary algorithms such as fuzzy logic and deep learning. Most of deep learning approches in inertial nvigation has focues on inertial odomotery and just few of them try to solve the inertial attitude estimation problem. Deep learning methods usually used for visual or visual-inertial based navigation. Chen et

Rochefort et al., proposed a neural networks-based satellite attitude estimation algorithm by using a quaternion neural network. This study presents a new way of integrating the neural network into the state estimator and develops a training procedure which is easy to implement. This algorithm provides the same accuracy as the EKF with significantly lower computational complexity. In [Chang 2011] a Time Varying Complementary Filter (TVCF) has been proped to use fuzzy logic inference system for CF parameters adjustment for the application of attitude estimation. Chen et al. deep recurrent neural networks for estimating the displacement of a user over a specified time window. OriNet [] intrduced by Esfahani et al., to estimate the orientation in quaternion form based on LSTM layers and IMU measuremetns.
[300] developed a sensor fusion method to provide pseudo-GPS position information by using empirical mode decomposition threshold filtering (EMDTF) for IMU noise elimination and a long short-term memory (LSTM) neural network for pseudo-GPS position predication during GPS outages.

Dhahbane et al. [301] developed a neural network-based complementary filter (NNCF) with ten hidden layers and trained by Bayesian Regularization Backpropagation (BRB) training algorithm to improve the generalization qualities and solve the overfitting problem. In this method output of complementary filter used as the neural network input.

Li et al., proposed an adaptive Kalman filter with a fuzzy neural network for trajectory estimation system mitigating the measurement noise and the undulation for the implementation of the touch interface.
An Adaptive Unscented Kalman Filter (AUKF) method  intrduced to combine sensor fusion algorithm with deep learning to achieve high precision attitude estimation based on low cost, small size IMU in high dynamic environment.
Deep Learing has been used in [] to denoise the gyroscope measuremetns for an open-loop attitude estimation algorithm.
Weber et al. [] present a real-time-capable neural network for robust IMU-based attitude estimation. In this study, accelerometer, gyrsocope, and IMU sampling rate has been used as input to the neural network and the output is the attitude in the quaternion form. This model only suitable for estimating the roll and pitch angle. Sun et al., intrduced a two-stage deep learning framwork for inertial odometry basd on LSTM and FFNN architcutre. In this study, the first stage is used to estimate the orientation and the second stage is used to estimate the position.
A Neural Network model has been developed by Santos et al. [] for static attitude determination based on PointNet architecture. They used attitude profile matrix as input. This model uses Swish activation function and Adam as its optimizer.

A deep learning model has been developed to estimate the Multirotor Unmanned Aerial Vehicle (MUAV) based on Kalman filter and Feed Forward Neural Network (FFNN) in []. LSTM framework has been used in [] the Euler angles using acceleromter, gyroscope and magnetometer but the sensor sampling rate has not been considered.

In the below table, we summarized some of the related works in the field of navigation using deep learning.

| Model             | Year/Month | Modality                    | Application                  |
| ----------------- | ---------- | --------------------------- | ---------------------------- |
| PoseNet           | 2015/12    | Vision                      | Relocalization               |
| VINet             | 2017/02    | Vision +Inertial          | Visual Inertial Odometry |
| DeepVO            | 2017/05    | Vision                      | Visual Odometry            |
| VidLoc            | 2017/07    | Vision                      | Relocalization               |
| PoseNet+          | 2017/07    | Vision                      | Relocalization               |
| SfmLearner        | 2017/07    | Vision                      | Visual Odometry            |
| IONet             | 2018/02    | Inertial Only             | Inertial Odometry          |
| UnDeepVO          | 2018/05    | Vision                      | Visual Odometry            |
| VLocNet           | 2018/05    | Vision                      | Relocalization, Odometry   |
| RIDI              | 2018/09    | Inertial Only             | Inertial Odometry          |
| SIDA              | 2019/01    | Inertial Only             | Domain Adaptation          |
| VIOLearner        | 2019/04    | Vision + Inertial         | Visual Inertial Odometry |
| Brossard et al.   | 2019/05    | Inertial Only             | Inertial Odometry          |
| SelectFusion      | 2019/06    | Vision + Inertial + LIDAR | VIO andSensor Fusion     |
| LO-Net            | 2019/06    | LIDAR                       | LIDAR Odometry             |
| L3-Net            | 2019/06    | LIDAR                       | LIDAR Odometry             |
| Lima et al.       | 2019/8     | Inertial                    | Inertial Odometry          |
| DeepVIO           | 2019/11    | Vision+Inertial             | Visual Inertial Odometry |
| OriNet            | 2020/4     | Inertial                    | Inertial Odometry            |
| GALNet            | 2020/5     | Inertial, Dynamic and Kinematic                    | Autonomous Cars   |
| PDRNet            | 2021/3     | Inertial                    | Pedestrian Dead Reckoning    |
| Kim et al.        | 2021/4     | Inertial                    | Inertial Odometry            |
| RIANN             | 2021/5     | Inertial                    | Attitude Estimation          |

## Problem definition

This study addressed the real time inertial attitude estimation based on gyroscope and accelerometer measuerments. The IMU sensor considered to rigidly attached to the object of interest. The estimaation is based on the current and pervious measurements of gyroscope and accelerometer which is used to fed into a Neural Network model to estimate the attitude. Despite almost all pervious studies, we do not consider any initial reset period for filter convergence. To aviod any singularites and have the least number of redundant parameters, we use quanternion representation with the componnets $[w, x, y, z]$ instead of Direction Cosine Matrix (DCM) or Euler angles. The error between the estimated attitude and the true attitude is calculated by quaternion multiplicative error and using the following equation:

$$
\begin{equation}
\begin{gathered}
\mathbf{q}_{err} = \mathbf{q}_{true} \otimes \mathbf{q}_{est}^{-1} 
\end{gathered}
\end{equation}
$$

where $\mathbf{q}_{err}$ represnet the shortest rotation between true and estimated orientation. The quaternion multiplication operator is calculated by the following equation:

$$
\begin{equation}
\begin{gathered}
\mathbf{q} \otimes \mathbf{p} = \begin{bmatrix}
q_0p_0 - q_1p_1 - q_2p_2 - q_3p_3 \\
q_0p_1 + q_1p_0 + q_2p_3 - q_3p_2 \\
q_0p_2 - q_1p_3 + q_2p_0 + q_3p_1 \\
q_0p_3 + q_1p_2 - q_2p_1 + q_3p_0
\end{bmatrix}
\end{gathered}
\end{equation}
$$

where $\mathbf{q}$ and $\mathbf{p}$ are the quanternions to be multiplied. The angle between the true and estimated orientation is calculated by the following equation:

$$
\begin{equation}
\begin{gathered}
\theta = 2 \arccos( scalar( \mathbf{q}_{err}) )
\end{gathered}
\end{equation}
$$

where $\theta$ is the angle between the true and estimated orientation. 



## Background

### Attitude

Attitude is the mathematical representation of the orientation in space related to the reference frames. Attitude parameters (attitude coordinates) refer to sets of parameters (coordinates) that fully describe a rigid body's attitude, which are not unique expressions. There are many ways to represent the attitude of a rigid body. The most common are the Euler angles, the rotation matrix, and the quaternions. The Euler angles are the most familiar form and known as yaw, pitch, and roll (or heading, elevation, and bank). Engineers widely use rotation matrix and quaternions, but the quaternions are less intuitive. The Euler angles are defined as the rotations about the three orthogonal axes of the body frame. But, the Euler angles suffer from the problem of gimbal lock. The rotation matrix is a 3x3 matrix that represents the orientation of the body frame with respect to the inertial frame which leads to have 6 redundant parameters. The quaternions are a 4x1 vector which are more suitable for attitude estimation because they are not subject to the gimbal lock problem and have the least redundant parameters. The quaternions are defined as the following:

$$
\begin{equation}
\begin{gathered}
\mathbf{q} =
\begin{bmatrix}
q_0 \\
q_1 \\
q_2 \\
q_3
\end{bmatrix}
\end{gathered}
\end{equation}
$$

where $q_0$ is the scalar part and $q_1$, $q_2$, and $q_3$ are the vector part. And the following equation shows the relationship between the quaternions and the euler angles:

$$
\begin{equation}
\begin{gathered}
\mathbf{q} =
\begin{bmatrix}
\cos(\phi/2) \cos(\theta/2) \cos(\psi/2) + \sin(\phi/2) \sin(\theta/2) \sin(\psi/2) \\
\sin(\phi/2) \cos(\theta/2) \cos(\psi/2) - \cos(\phi/2) \sin(\theta/2) \sin(\psi/2) \\
\cos(\phi/2) \sin(\theta/2) \cos(\psi/2) + \sin(\phi/2) \cos(\theta/2) \sin(\psi/2) \\
\cos(\phi/2) \cos(\theta/2) \sin(\psi/2) - \sin(\phi/2) \sin(\theta/2) \cos(\psi/2)
\end{bmatrix}
\end{gathered}
\end{equation}
$$

where $\phi$, $\theta$, and $\psi$ are the Euler angles.

Attitude determination and control play a vital role in Aerospace engineering. Most aerial or space vehicles have subsystem(s) that must be pointed to a specific direction, known as pointing modes, e.g., Sun pointing, Earth pointing. For example, communications satellites, keeping satellites antenna pointed to the Earth continuously, is the key to the successful mission. That will be achieved only if we have proper knowledge of the vehicle’s orientation; in other words, the attitude must be determined. Attitude determination methods can be divided in two categories: static and dynamic.

Static attitude determination is a point-to-point time independent attitude determining method with the memoryless approach is called attitude determination. It is the observations or measurements processing to obtain the information for describing the object's orientation relative to a reference frame. It could be determined by measuring the directions from the vehicle to the known points, i.e., Attitude Knowledge. Due to accuracy limit, measurement noise, model error, and process error, most deterministic approaches are inefficient for accurate prospects; in this situation, using statistical methods will be a good solution

Dynamic attitude determination methods also known as Attitude estimation refers to using mathematical methods and techniques (e.g., statistical and probabilistic) to predict and estimate the future attitude based on a dynamic model and prior measurements. These techniques fuse data that retain a series of measurements using algorithms such as filtering, Multi-Sensor-Data-Fusion. The most commonly use attitude estimation methods are Extended Kalman Filter, Madgwick, and Mahony.

### Attitude Determination from Inertial Sensors

Attitude could be measured based on accelerometer and gyroscope readings. Gyroscope meaesures the angular velocity in body frame about the three orthogonal axes (i.e., x,y,z) usually denotd by $p$, $q$, and $r$ and relays on the principle of the angular momentum conservation. The gyroscope output, body rates with respect to the inertial frame which expressed in body frame is:

$$
\begin{equation}
\begin{gathered}
\mathbf{\omega} =
\begin{bmatrix}
\omega_x \\
\omega_y \\
\omega_z
\end{bmatrix}
\end{gathered}
\end{equation}
$$

where $\omega_x$, $\omega_y$, and $\omega_z$ are the angular velocity about the x, y, and z axes, respectively. The accelerometer measures the linear acceleration in body frame about the three orthogonal axes (i.e., x,y,z) usually denotd by $a_x$, $a_y$, and $a_z$ and relays on the principle of Newton's second law. The accelerometer output, linear acceleration with respect to the inertial frame which expressed in body frame is:

$$
\begin{equation}
\begin{gathered}
\mathbf{a} =
\begin{bmatrix}
a_x \\
a_y \\
a_z
\end{bmatrix}
\end{gathered}
\end{equation}
$$

Attitude can be determined from the accelerometer and gyroscope readings using the following equations:

$$
\begin{equation}
\begin{gathered}
\phi = \arctan\left(\frac{a_y}{a_z}\right) \\
\theta = \arctan\left(\frac{-a_x}{\sqrt{a_y^2 + a_z^2}}\right) \\
\end{gathered}
\end{equation}
$$

Attitude update using gyroscope readings:

$$
\begin{equation}
\begin{gathered}
\dot{\phi} = p + q \sin(\phi) \tan(\theta) + r \cos(\phi) \tan(\theta) \\
\dot{\theta} = q \cos(\phi) - r \sin(\phi) \\
\dot{\psi} = \frac{q \sin(\phi)}{\cos(\theta)} + \frac{r \cos(\phi)}{\cos(\theta)} \\
\end{gathered}
\end{equation}
$$

where $\phi$, $\theta$, and $\psi$ are the Euler angles. Or in the quaternion form:

$$
\begin{equation}
\begin{gathered}
\mathbf{\dot{q}} = \frac{1}{2} \mathbf{q} \otimes \mathbf{\omega}
\end{gathered}
\end{equation}
$$

where $\mathbf{\dot{q}}$ is the quaternion derivative, $\mathbf{q}$ is the quaternion, and $\mathbf{\omega}$ is the angular velocity. It is necessary to mention that heading angle $\psi$ is not determined from the accelerometer, and gyroscope readings only can be used to measure the rate of change of the heading angle.



## Methodology

### Deep Learning Model

### Loss Function

## Experiment

### Dataset
