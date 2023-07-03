---
title: "CNN Based Attitude Estimation"
date: 2022-10-25
url: /Attitude-Estimation-CNN/
description: "" 
summary: "" 
showToc: true
math: true
disableAnchoredHeadings: false
commentable: true
tags:
  - Navigation
  - Attitude Estimation
  - Orientaion Estimation
---

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Problem definition](#problem-definition)
- [Background](#background)
  - [Attitude](#attitude)
  - [Attitude Determination from Inertial Sensors](#attitude-determination-from-inertial-sensors)
- [Methodology](#methodology)
  - [Deep Learning Model](#deep-learning-model)
  - [Loss Function](#loss-function)
- [Experiment](#experiment)
  - [Dataset](#dataset)
- [RepoIMU T-stick](#repoimu-t-stick)
- [RepoIMU T-pendulum](#repoimu-t-pendulum)
- [Sassari](#sassari)
- [OxIOD](#oxiod)
- [MAV Dataset](#mav-dataset)
- [EuRoC MAV](#euroc-mav)
- [TUM-VI](#tum-vi)
- [KITTI](#kitti)
- [RIDI](#ridi)
- [RoNIN](#ronin)
- [BROAD](#broad)
- [Training](#training)
- [Evaluation](#evaluation)


## Abstract

This article discusses the importance of accurate and precise attitude determination in navigation for air and space vehicles. Various instruments and sensors have been developed over the last few decades to achieve this goal. However, the cost and complexity of these instruments can be prohibitive. To address this issue, Multi-Data Sensor Fusion (MSDF) techniques have been developed, which allow for the use of multiple sensors to sense a quantity from different perspectives or sense multiple quantities to reduce errors and uncertainties. This article explores the use of MEMS-based Inertial Measurement Units (IMUs) in attitude determination and discusses the challenges associated with noise and bias. Finally, the article describes different forms of attitude representation, including Tait-Bryan angles, rotation matrices, and quaternions.

## Introduction

Achieving accurate and precise attitude determination or estimation is essential for successful navigation of air and space vehicles. To achieve this goal, each vehicle must determine and control its attitude based on mission requirements. A wide variety of instruments, sensors, and algorithms have been developed over the last few decades, distinguished by their cost and complexity. However, using an accurate sensor can exponentially increase the cost, which may exceed the budget.

A low-cost solution for achieving high accuracy is to use multiple sensors (homogeneous or heterogeneous) to sense a quantity from different perspectives or sense multiple quantities to reduce errors and uncertainties. Multiple sensors fuse their data to achieve a more accurate quantity, a technique called Multi-Data Sensor Fusion (MSDF). MSDF uses mathematical methods to reduce noise and uncertainty and to estimate the quantity based on prior data. MSDF can also be used for attitude determination.

Attitude determination methods can be broadly divided into two classes: single-point and recursive estimation. The first method calculates the attitude by using two or more vector measurements at a single point in time. In contrast, recursive methods use the combination of measurements over time and the system's mathematical model. Obtaining precise attitude determination is challenging due to errors in system modeling, processes, and measurements. Increasing the sensor's precision may exponentially increase the cost, and sometimes, achieving the required precision may only be possible at an exorbitant cost.

One approach for determining attitude is to use inertial navigation algorithms based on inertial sensors. Inertial navigation is based on the Dead Reckoning method, which uses different types of inertial sensors, such as accelerometers and gyroscopes, known as Inertial Measurement Units (IMUs). The position, velocity, and attitude of a moving object can be determined using numerical integration of IMU measurements.

The use of low-cost Micro Electro Mechanical Systems (MEMS) based IMUs has grown in the past decade. Due to recent advances in MEMS technology, IMUs have become smaller, cheaper, and more accurate, making them available for use in mobile robots, smartphones, drones, and autonomous vehicles. However, these sensors suffer from noise and bias, which directly affect the performance of attitude estimation algorithms.

To tackle this problem and increase the accuracy and reliability of attitude estimation techniques, different MSDF techniques and Deep Learning models have been developed in the past few decades.

Attitude can be represented in many different forms. The Tait-Bryan angles (also called Euler angles) are the most familiar form and are known as yaw, pitch, and roll (or heading, elevation, and bank). Engineers widely use rotation matrices and quaternions, but quaternions are less intuitive.

Related Work
In the past decade, extensive research has been conducted on inertial navigation techniques. These studies can be roughly divided into three categories: estimation methods, Multi-Data Sensor Fusion (MSDF) techniques, and evolutionary/AI algorithms. The Kalman Filter family (i.e., EKF, UKF, MEKF), as well as other commonly used algorithms such as Madgwick and Mahony, are based on the dynamic model of the system. The Kalman filter was first introduced in [], and its variants such as EKF, UKF, and MEKF have been implemented for attitude estimation applications.

In [], Carsuo et al. compared different sensor fusion algorithms for inertial attitude estimation. This comparative study showed that the performance of Sensor Fusion Algorithms (SFA) is highly dependent on parameter tuning, and fixed parameter values are not suitable for all applications. Therefore, parameter tuning is one of the disadvantages of conventional attitude estimation methods. This problem could be tackled by using evolutionary algorithms such as fuzzy logic and deep learning. Most of the deep learning approaches in inertial navigation have focused on inertial odometry, and only a few of them have attempted to solve the inertial attitude estimation problem. Deep learning methods are usually used for visual or visual-inertial based navigation. Chen et al.,

Rochefort et al. proposed a neural networks-based satellite attitude estimation algorithm using a quaternion neural network. This study presents a new way of integrating the neural network into the state estimator and develops a training procedure that is easy to implement. This algorithm provides the same accuracy as the EKF with significantly lower computational complexity. In [Chang 2011], a Time-Varying Complementary Filter (TVCF) has been proposed to use a fuzzy logic inference system for CF parameter adjustment for attitude estimation. Chen et al. used deep recurrent neural networks for estimating the displacement of a user over a specified time window. OriNet [], introduced by Esfahani et al., estimates the orientation in quaternion form based on LSTM layers and IMU measurements.

[300] developed a sensor fusion method to provide pseudo-GPS position information by using empirical mode decomposition threshold filtering (EMDTF) for IMU noise elimination and a long short-term memory (LSTM) neural network for pseudo-GPS position prediction during GPS outages.

Dhahbane et al. [301] developed a neural network-based complementary filter (NNCF) with ten hidden layers and trained by Bayesian Regularization Backpropagation (BRB) training algorithm to improve the generalization qualities and solve the overfitting problem. In this method, the output of the complementary filter is used as the neural network input.

Li et al. proposed an adaptive Kalman filter with a fuzzy neural network for a trajectory estimation system that mitigates measurement noise and undulation for the implementation of the touch interface. An Adaptive Unscented Kalman Filter (AUKF) method was introduced to combine sensor fusion algorithms with deep learning to achieve high-precision attitude estimation based on low-cost, small size IMU in high dynamic environments. Deep Learning has been used in [] to denoise the gyroscope measurements for an open-loop attitude estimation algorithm.

Weber et al. [] present a real-time-capable neural network for robust IMU-based attitude estimation. In this study, the accelerometer, gyroscope, and IMU sampling rate were used as inputs to the neural network, and the output is the attitude in quaternion form. This model is only suitable for estimating the roll and pitch angle. Sun et al. introduced a two-stage deep learning framework for inertial odometry based on LSTM and FFNN architecture. In this study, the first stage is used to estimate the orientation, and the second stage is used to estimate the position.


A Neural Network model has been developed by Santos et al. [] for static attitude determination based on PointNet architecture. They used attitude profile matrix as input. This model uses Swish activation function and Adam as its optimizer.

A deep learning model has been developed to estimate the Multirotor Unmanned Aerial Vehicle (MUAV) based on Kalman filter and Feed Forward Neural Network (FFNN) in []. LSTM framework has been used in [] the Euler angles using acceleromter, gyroscope and magnetometer but the sensor sampling rate has not been considered.

In the below table, we summarized some of the related works in the field of navigation using deep learning.

| Model           | Year/Month | Modality                        | Application               |
| --------------- | ---------- | ------------------------------- | ------------------------- |
| PoseNet         | 2015/12    | Vision                          | Relocalization            |
| VINet           | 2017/02    | Vision +Inertial                | Visual Inertial Odometry  |
| DeepVO          | 2017/05    | Vision                          | Visual Odometry           |
| VidLoc          | 2017/07    | Vision                          | Relocalization            |
| PoseNet+        | 2017/07    | Vision                          | Relocalization            |
| SfmLearner      | 2017/07    | Vision                          | Visual Odometry           |
| IONet           | 2018/02    | Inertial Only                   | Inertial Odometry         |
| UnDeepVO        | 2018/05    | Vision                          | Visual Odometry           |
| VLocNet         | 2018/05    | Vision                          | Relocalization, Odometry  |
| RIDI            | 2018/09    | Inertial Only                   | Inertial Odometry         |
| SIDA            | 2019/01    | Inertial Only                   | Domain Adaptation         |
| VIOLearner      | 2019/04    | Vision + Inertial               | Visual Inertial Odometry  |
| Brossard et al. | 2019/05    | Inertial Only                   | Inertial Odometry         |
| SelectFusion    | 2019/06    | Vision + Inertial + LIDAR       | VIO andSensor Fusion      |
| LO-Net          | 2019/06    | LIDAR                           | LIDAR Odometry            |
| L3-Net          | 2019/06    | LIDAR                           | LIDAR Odometry            |
| Lima et al.     | 2019/8     | Inertial                        | Inertial Odometry         |
| DeepVIO         | 2019/11    | Vision+Inertial                 | Visual Inertial Odometry  |
| OriNet          | 2020/4     | Inertial                        | Inertial Odometry         |
| GALNet          | 2020/5     | Inertial, Dynamic and Kinematic | Autonomous Cars           |
| PDRNet          | 2021/3     | Inertial                        | Pedestrian Dead Reckoning |
| Kim et al.      | 2021/4     | Inertial                        | Inertial Odometry         |
| RIANN           | 2021/5     | Inertial                        | Attitude Estimation       |

## Problem definition

This study addressed the real time inertial attitude estimation based on gyroscope and accelerometer measuerments. The IMU sensor considered to rigidly attached to the object of interest. The estimaation is based on the current and pervious measurements of gyroscope and accelerometer which is used to fed into a Neural Network model to estimate the attitude. Despite almost all pervious studies, we do not consider any initial reset period for filter convergence. Usually, to aviod any singularites and have the least number of redundant parameters, quanternion representation with the componnets $[w, x, y, z]$ is used, instead of Direction Cosine Matrix (DCM) or Euler angles. But as the angles have different features and dependencies to the sensor readings, we convert quaternions to Euler angles and tried to estimate the roll and pitch based on the accelerometer and gyroscope readings. The quaternions could be converted to Euler angles using the following equations:

<div>
$$
\begin{equation}
\begin{gathered}
\phi = \arctan \left(\frac{2 \left(q_{w} q_{x} + q_{y} q_{z}\right)}{1 - 2 \left(q_{x}^{2} + q_{y}^{2}\right)}\right) \\
\end{gathered}
\end{equation}
$$
</div><div>
$$
\begin{equation}
\begin{gathered}
\theta = \arcsin \left(2 \left(q_{w} q_{y} - q_{z} q_{x}\right)\right) \\
\end{gathered}
\end{equation}
$$
</div>

## Background

### Attitude

Attitude is the mathematical representation of the orientation in space related to the reference frames. Attitude parameters (attitude coordinates) refer to sets of parameters (coordinates) that fully describe a rigid body's attitude, which are not unique expressions. There are many ways to represent the attitude of a rigid body. The most common are the Euler angles, the rotation matrix, and the quaternions. The Euler angles are the most familiar form and known as yaw, pitch, and roll (or heading, elevation, and bank). Engineers widely use rotation matrix and quaternions, but the quaternions are less intuitive. The Euler angles are defined as the rotations about the three orthogonal axes of the body frame. But, the Euler angles suffer from the problem of gimbal lock. The rotation matrix is a 3x3 matrix that represents the orientation of the body frame with respect to the inertial frame which leads to have 6 redundant parameters. The quaternions are a 4x1 vector which are more suitable for attitude estimation because they are not subject to the gimbal lock problem and have the least redundant parameters. The quaternions are defined as the following:

<div>
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
<div>

where $q_0$ is the scalar part and $q_1$, $q_2$, and $q_3$ are the vector part. And the following equation shows the relationship between the quaternions and the euler angles:

<div>
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
</div>

where $\phi$, $\theta$, and $\psi$ are the Euler angles.

Attitude determination and control play a vital role in Aerospace engineering. Most aerial or space vehicles have subsystem(s) that must be pointed to a specific direction, known as pointing modes, e.g., Sun pointing, Earth pointing. For example, communications satellites, keeping satellites antenna pointed to the Earth continuously, is the key to the successful mission. That will be achieved only if we have proper knowledge of the vehicle’s orientation; in other words, the attitude must be determined. Attitude determination methods can be divided in two categories: static and dynamic.

Static attitude determination is a point-to-point time independent attitude determining method with the memoryless approach is called attitude determination. It is the observations or measurements processing to obtain the information for describing the object's orientation relative to a reference frame. It could be determined by measuring the directions from the vehicle to the known points, i.e., Attitude Knowledge. Due to accuracy limit, measurement noise, model error, and process error, most deterministic approaches are inefficient for accurate prospects; in this situation, using statistical methods will be a good solution

Dynamic attitude determination methods also known as Attitude estimation refers to using mathematical methods and techniques (e.g., statistical and probabilistic) to predict and estimate the future attitude based on a dynamic model and prior measurements. These techniques fuse data that retain a series of measurements using algorithms such as filtering, Multi-Sensor-Data-Fusion. The most commonly use attitude estimation methods are Extended Kalman Filter, Madgwick, and Mahony.

### Attitude Determination from Inertial Sensors

Attitude could be measured based on accelerometer and gyroscope readings. Gyroscope meaesures the angular velocity in body frame about the three orthogonal axes (i.e., x,y,z) usually denotd by $p$, $q$, and $r$ and relays on the principle of the angular momentum conservation. The gyroscope output, body rates with respect to the inertial frame which expressed in body frame is:

<div>
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
</div>

where $\omega_x$, $\omega_y$, and $\omega_z$ are the angular velocity about the x, y, and z axes, respectively. The accelerometer measures the linear acceleration in body frame about the three orthogonal axes (i.e., x,y,z) usually denotd by $a_x$, $a_y$, and $a_z$ and relays on the principle of Newton's second law. The accelerometer output, linear acceleration with respect to the inertial frame which expressed in body frame is:

<div>
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
</div>

Attitude can be determined from the accelerometer and gyroscope readings using the following equations:

<div>
$$
\begin{equation}
\begin{gathered}
\phi = \arctan\left(\frac{a_y}{a_z}\right) \\
\theta = \arctan\left(\frac{-a_x}{\sqrt{a_y^2 + a_z^2}}\right) \\
\end{gathered}
\end{equation}
$$
</div>

Attitude update using gyroscope readings:

<div>
$$
\begin{equation}
\begin{gathered}
\dot{\phi} = p + q \sin(\phi) \tan(\theta) + r \cos(\phi) \tan(\theta) \\
\dot{\theta} = q \cos(\phi) - r \sin(\phi) \\
\dot{\psi} = \frac{q \sin(\phi)}{\cos(\theta)} + \frac{r \cos(\phi)}{\cos(\theta)} \\
\end{gathered}
\end{equation}
$$
</div>

where $\phi$, $\theta$, and $\psi$ are the Euler angles. Or in the quaternion form:

<div>
$$
\begin{equation}
\begin{gathered}
\mathbf{\dot{q}} = \frac{1}{2} \mathbf{q} \otimes \mathbf{\omega}
\end{gathered}
\end{equation}
$$
</div>

where $\mathbf{\dot{q}}$ is the quaternion derivative, $\mathbf{q}$ is the quaternion, and $\mathbf{\omega}$ is the angular velocity. It is necessary to mention that heading angle $\psi$ is not determined from the accelerometer, and gyroscope readings only can be used to measure the rate of change of the heading angle.

## Methodology

An eficiant way to handel the sequnetial data such as IMU sensor measurements is to use seqential modeling. This type of modeling can be used to carry out time series data. In this project, we will use the Long Short-Term Memory (LSTM) network to model the sequential data. The LSTM network is a type of recurrent neural network (RNN) that is capable of learning order dependence in sequence prediction problems. It is a complex network of artificial neurons, arranged in a long chain. Each unit in the chain contains a memory cell that has three gates: input gate, forget gate, and output gate. The LSTM network is trained using backpropagation through time and overcomes the vanishing gradient problem. The LSTM network is able to learn long-term dependencies and is therefore very well suited to predict the next value in a sequence.

### Deep Learning Model

The proposed method for attitude estimation based on inertial measurements, takes a sequence of accelerometer and gyroscope readings and its corexponding time stamps as an input, and outputs roll and pitch angles. This end-to-end deep learning framework, implicitly handels the IMU measruments noise and bias. This solution is based on CNN layers combined with LSTM layers. The CNN layers are used to extract the features from the accelerometer and gyroscope readings, and the LSTM layers are used to learn the temporal dependencies between the extracted features. The input to the network is a sequence of accelerometer and gyroscope readings in a window of 200 readings. Accelerometer and gyroscope measurements are processed seprately by 1-dimentional CNN layers with the kernel size of 11 and 128 filters. The output of the CNN layers are concatinated and fed to the LSTM layer after max pooling layer of size 3. This bi-stack LSTM layer has 128 unit. The output of this layer seprately fed to two LSTM layer, each with 128 units and the outputs are fed to two fully connected layers with 1 units. The output of the fully connected layers are the roll and pitch angles. After each LSTM layer, a dropout layer with 0.25 probability is added to prevent overfitting. This layer is used to randomly drop out 25% of the units in the layer during training. The input in each timp step is a window of 200 accelerometer and gyroscope readings which consists of 100 past and 100 future readings. The stride of window is 10 frames which led the model to estimate the attitud every 10 frames. The network is trained using the Adam optimizer with the learning rate of [lr] and the loss function is the mean squared error. The network is trained on the BROAD and OxIOD datasets for [epochs] epochs with the batch size of [batch_size]. The network is implemented using the Keras library with the TensorFlow backend.

### Loss Function

## Experiment

### Dataset

IMU Datasets are used to evaluate the performance of the attitude estimation algorithms. The datasets are divided into two categories: synthetic and real-world. The synthetic datasets are generated by simulating the IMU measurements. The real-world datasets are collected from the real-world experiments and could be divided into two categories: indoor and outdoor. The indoor experiments are conducted in a controlled environment, e.g., a laboratory. The outdoor experiments are conducted in an uncontrolled environment, e.g., a car. Also, to train, validate and test any neural network model, we need a database including accurate input and output. A Deep Learning model's performance will be directly affected by the data that is used for it. So, to train the Deep Learning model we need a database containing the input and output parameters with following conditions:

- The input and output parameters should be accurate.
- The amount of data must be sufficient to train the Deep Learning model
- The data should be diverse enough to cover all the possible scenarios.

In the following sections, we will present some of the most commonly used IMU Datasets.

## RepoIMU T-stick

The RepoIMU T-stick [<a id="d1" href="#repoT">1</a>] is a small, low-cost, and high-performance inertial measurement unit (IMU) that can be used for a wide range of applications. The RepoIMU T-stick is a 9-axis IMU that measures the acceleration, angular velocity, and magnetic field. This database contains two separate sets of experiments recorded with a T-stick and a pendulum. A total of 29 trials were collected on the T-stick, and each trial lasted approximately 90 seconds. As the name suggests, the IMU is attached to a T-shaped stick equipped with six reflective markers. Each experiment consists of slow or fast rotation around a principal sensor axis or translation along a principal sensor axis. In this scenario, the data from the Vicon Nexus OMC system and the XSens MTi IMU are synchronized and provided at a frequency of 100 Hz. The authors clearly state that the IMU coordinate system and the ground trace are not aligned and propose a method to compensate for one of the two required rotations based on quaternion averaging. Unfortunately, some experiments contain gyroscope clipping and ground tracking, which significantly affect the obtained errors. Therefore, careful pre-processing and removal of some trials should be considered when using the dataset to evaluate the model's accuracy. The dataset is available at <a href="http://https://github.com/agnieszkaszczesna/RepoIMU">Link</a>.

## RepoIMU T-pendulum

The second part of the RepoIMU dataset contains data from a triple pendulum on which the IMUs are mounted. Measurement data is provided at 90 Hz or 166 Hz. However, the IMU data contains duplicate samples. This is usually the result of artificial sampling or transmission problems where missed samples are replaced by duplicating the last sample received, effectively reducing the sampling rate. The sampling rate achieved when discarding frequent samples is about 25 Hz and 48 Hz for the accelerometer and gyroscope, respectively. Due to this issue, it is not recommended to use this database for model training and evaluation. Due to this fact, we cannot recommend using pendulum tests to evaluate the accuracy of IOE with high precision.

## Sassari

The Sassari dataset published in [<a id="d2" href="#sassari">2</a>] aims to validate a parameter tuning approach based on the orientation difference of two IMUs of the same model. To facilitate this, six IMUs from three manufacturers (Xsens, APDM, Shimmer) are placed on a wooden board. Rotation around specific axes and free rotation around all axes are repeated at three different speeds. Data is synchronized and presented at 100 Hz. Local coordinate frames are aligned by precise manual placement. There are 18 experiments (3 speeds, 3 IMU models, and 2 IMUs of each model) in this dataset.

According to these points, this database seems to be a suitable option for training, evaluating, and testing the model, but some essential points should be paid attention to. The inclusion of different speeds and different types of IMUs helped to diversify the data set. However, all motions occur in a homogeneous magnetic field and do not involve pure translational motions. Therefore, this data set does not have a robust variety in terms of the type of movement and the variety of magnetic data. Therefore, the model trained with it cannot be robust and general. However, it can be used to evaluate the model.

The total movement duration of all three trials is 168 seconds, with the most extended movement phase lasting 30 seconds. For this reason, considering the short time, it is not a suitable option for training. The dataset is available at <a href="http://https://ieee-dataport.org/documents/mimuopticalsassaridataset">Link</a>.

## OxIOD

The Oxford Inertial Odometry Dataset (OxIOD) [<a id="d3" href="#oxiod">3</a>] is a large set of inertial data recorded by smartphones (mainly iPhone 7 Plus) at 100 Hz. The suite consists of 158 tests and covers a distance of over 42 km, with OMC ground track available for 132 tests. The purpose of this set is inertial odometry. Therefore, it does not include pure rotational movements and pure translational movements, which are helpful for systematically evaluating the model's performance under different conditions; however, it covers a wide range of everyday movements.

Due to the different focus, some information (for example, the alignment of the coordinate frames) is not accurately described. In addition, the orientation of the ground trace contains frequent irregularities (e.g., jumps in orientation that are not accompanied by similar jumps in the IMU data). The dataset is available at <a href="http://deepio.cs.ox.ac.uk/">Link</a>.

## MAV Dataset

Most datasets suitable for the simultaneous localization and mapping problem are collected from sensors such as wheel encoders and laser range finders mounted on ground robots. For small air vehicles, there are few datasets, and MAV Dataset [<a id="d4" href="#mav">4</a>] is one of them. This data set was collected from the sensor array installed on the "Pelican" quadrotor platform in an environment. The sensor suite includes a forward-facing camera, a downward-facing camera, an inertial measurement unit, and a Vicon ground-tracking system. Five synchronized datasets are presented

<ol>
<li> 1LoopDown </li>
<li> 2LoopsDown </li>
<li> 3LoopsDown </li>
<li> hoveringDown </li>
<li> randomFront </li>
</ol>

These datasets include camera images, accelerations, heading rates, absolute angles from the IMU, and ground tracking from the Vicon system. The dataset is available at <a href="https://sites.google.com/site/sflyorg/mav-datasets">Link</a>.

## EuRoC MAV

The EuRoC MAV dataset [<a id="d5" href="#euroc">5</a>] is a large dataset collected from a quadrotor MAV. The dataset contains the internal flight data of a small air vehicle (MAV) and is designed to reconstruct the visual-inertial 3D environment. The six experiments performed in the chamber and synchronized and aligned using the OMC-based Vicon ground probe are suitable for training and evaluating the model's accuracy. It should be noted that camera images and point clouds are also included.

This set does not include magnetometer data, which limits the evaluation of three degrees of freedom and is only for two-way models (including accelerometer and gyroscope). Due to the nature of the data, most of the movement consists of horizontal transfer and rotation around the vertical axis. This slope does not change much during the experiments. For this reason, it does not have a suitable variety for model training. Since flight-induced vibrations are clearly visible in the raw accelerometer data, the EuRoC MAV dataset provides a unique test case for orientation estimation with perturbed accelerometer data. The dataset is available at <a href="https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets">Link</a>.

## TUM-VI

The TUM Visual-Inertial Dataset [<a id="d6" href="#tumvi">6</a>] suitable for optical-inertial odometry consists of 28 experiments with a handheld instrument equipped with a camera and IMU. Due to this application focus, most experiments only include OMC ground trace data at the beginning and at the end of the experiment. However, the six-chamber experiments include complete OMC data. They are suitable for evaluating the accuracy of the neural network model. Similar to the EuRoC MAV data, the motion consists mainly of horizontal translation and rotation about the vertical axis, and magnetometer data is not included. The dataset is available at <a href="https://vision.in.tum.de/data/datasets/visual-inertial-dataset">Link</a>.

## KITTI

The KITTI Vision Benchmark Suite [<a id="d7" href="#kitti">7</a>] is a large set of data collected from a stereo camera and a laser range finder mounted on a car. The dataset includes 11 sequences with a total of 20,000 images. The dataset is suitable for evaluating the accuracy of the model in the presence of optical flow. However, the dataset does not include magnetometer data, which limits the evaluation of three degrees of freedom and is only for two-way models (including accelerometer and gyroscope). The dataset is available at <a href="http://www.cvlibs.net/datasets/kitti/eval_odometry.php">Link</a>.

## RIDI

RIDI datasets were collected over 2.5 hours on 10 human subjects using smartphones equipped with a 3D tracking capability to collect IMU-motion data placed on four different surfaces (e.g., the hand, the bag, the leg pocket, and the body). The ground-truth motion data was produced by the Visual Inertial SLAM technique. They recorded linear accelerations, angular velocities, gravity directions, device orientations (via Android APIs), and 3D camera poses with a Google Tango phone, Lenovo Phab2 Pro. Visual Inertial Odometry on Tango provides camera poses that are accurate enough for inertial odometry purposes (less than 1 meter after 200 meters of tracking).The dataset is available at <a href="http://www.cvlibs.net/datasets/kitti/eval_odometry.php">Link</a>.

## RoNIN

The RoNIN dataset [<a id="d9" href="#ridi">9</a>] contains over 40 hours of IMU sensor data from 100 human subjects with 3D ground-truth trajectories under natural human movements. This data set provides measurements of the accelerometer, gyroscope, dipstick, GPS, and ground track, including direction and location in 327 sequences and at a frequency of 200 Hz. A two-device data collection protocol was developed. A harness was used to attach one phone to the body for 3D tracking, allowing subjects to control the other phone to collect IMU data freely. It should be noted that the ground track can only be obtained using the 3D tracker phone attached to the harness. In addition, the body trajectory is estimated instead of the IMU. The dataset is available at <a href="https://yanhangpublic.github.io/ridi/index.html">Link</a>.

## BROAD

The Berlin Robust Orientation Evaluation (BROAD) dataset [<a id="d10" href="#broad">10</a>] includes a diverse set of experiments covering a variety of motion types, velocities, undisturbed motions, and motions with intentional accelerometer perturbations as well as motions performed in the presence of magnetic perturbations. This data set includes 39 experiments (23 undisturbed experiments with different movement types and speeds and 16 experiments with various intentional disturbances). The data of the accelerometer, gyroscope, magnetometer, quaternion, and ground tracks, are provided in an ENU frame with a frequency of 286.3 Hz. The dataset is available at <a href="https://github.com/dlaidig/broad">Link</a>.

Based on datasets preprocessing requierments, diversity of motion types, and the availability of ground truth, we selected the BROAD, OxIOD, RIDI, and RoNIN datasets for our experiments and analysis. This combination of datasets provides a wide spectrum of motion types and speeds, as well as the presence of intentional disturbances such as vibration and acceleration. Also, as each dataset has its own sampling frequency, it led us to  train our model on different sampling frequencies which is a key factor for sampling rate robustness. These datasets are come from various applications and motion patterns, which makes them suitable for evaluating the accuracy of the model in different scenarios. The details of the datasets are summarized in Table [1]. Figure [1] shows the collection of the datasets composed of the BROAD, OxIOD, RIDI, and RoNIN datasets which are splited into, training, validation, and testing sets. The validation dataset is used to evaluate the model during training, and the test dataset is used to evaluate the model performace after training.


## Training

We implementede this model in Keras [v] and Tensorflow [v], also Adam optimizer was used for training with learning rate starts from 0.0001 and controled via ReduceLROnPlateau. The model was trained on a single NVIDIA GeForce GTX 1070 GPU. The model was trained for 100 epochs with a batch size of 500. The training data has been shuffled and split into 75% training and 25% validation. The training and validation loss and accuracy are shown in the following figure. Using EarlyStopping callback, help us to stop the training when no improvement in the validation loss is observed. In addition, checkpoints are saved during training to restore the model to the best validation loss and ensure that the best model is saved. 

## Evaluation

For performance evaluation, we compared the proposed model with RIANN model, EKF, Madgwick, Mahony, and Complementary filter. The results of the proposed model are shown in the following figure. The performance of the propesd model shows that it could be considered as a good alternative to the state-of-the-art methods and convetional filters as the model performed well on a wide range of motion types, patterns, and speeds. While the convetional filters require parameter optimiztion for each motion type 