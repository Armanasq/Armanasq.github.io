---
title: "KITTI Dataset"
date: 2023-04-01
description: "Introduction"
url: "/kitti/"
showToc: true
meta:
  - property: og:locale
    content: en-US
  - property: og:site_name
    content: "Arman Asgharpoor"
  - property: og:title
    content: "KITTI Dataset"
  - property: og:url
    content: "http://localhost:4000/datasets/kitti/"
  - property: og:description
    content: "Introduction"
  - name: "HandheldFriendly"
    content: "True"
  - name: "MobileOptimized"
    content: "320"
  - name: "viewport"
    content: "width=device-width, initial-scale=1.0"
  - http-equiv: "cleartype"
    content: "on"
output:
  html_document:
    css: http://localhost:4000/assets/css/main.css
---

<p><img src="/images/kitti-cover.png" alt="KITTI Datasets" /></p>

## Introduction

<p>KITTI is a popular computer vision dataset designed for autonomous driving research. It contains a diverse set of challenges for researchers, including object detection, tracking, and scene understanding. The dataset is derived from the autonomous driving platform developed by the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago.</p>

<p>The KITTI dataset includes a collection of different sensors and modalities, such as stereo cameras, LiDAR, and GPS/INS sensors, which provides a comprehensive view of the environment around the vehicle. The data was collected over several days in the urban areas of Karlsruhe and nearby towns in Germany. The dataset includes more than 200,000 stereo images and their corresponding point clouds, as well as data from the GPS/INS sensors, which provide accurate location and pose information.</p>

<p>The dataset is divided into several different categories, each with its own set of challenges. These categories include object detection, tracking, scene understanding, visual odometry, and road/lane detection. Each category contains a set of challenges that researchers can use to evaluate their algorithms and compare their results with others in the field.</p>

<p>One of the strengths of the KITTI dataset is its accuracy and precision. The sensors used to collect the data provide a high level of detail and accuracy, making it possible to detect and track objects with high precision. Additionally, the dataset includes a large number of real-world scenarios, which makes it more representative of real-world driving conditions.</p>

<p>Another strength of the KITTI dataset is its large size. The dataset includes over 50 GB of data, which includes stereo images, point clouds, and GPS/INS data. This large amount of data makes it possible to train deep neural networks, which are known to perform well on large datasets.</p>

<p>Despite its strengths, the KITTI dataset also has some limitations. For example, the dataset only covers urban driving scenarios, which may not be representative of driving conditions in other environments. Additionally, the dataset is relatively small compared to other computer vision datasets, which may limit its applicability in certain domains.</p>

<p>In summary, the KITTI dataset is a valuable resource for researchers in the field of autonomous driving. Its accuracy, precision, and large size make it an ideal dataset for evaluating and comparing algorithms for object detection, tracking, and scene understanding. While it has some limitations, its strengths make it a popular and widely used dataset in the field.</p>

## Data Format

<p>The KITTI dataset is available in two formats: raw data and preprocessed data. The raw data contains a large amount of sensor data, including images, LiDAR point clouds, and GPS/IMU measurements, and can be used for various research purposes. The preprocessed data provides more structured data, including object labels, and can be used directly for tasks such as object detection and tracking.</p>

## Downloading the Dataset

<p>The KITTI dataset can be downloaded from the official website (http://www.cvlibs.net/datasets/kitti/) after registering with a valid email address. The website provides instructions on how to download and use the data, including the required software tools and file formats.</p>

## Using the KITTI Dataset in Python

### Prerequisites

<p>Before getting started, make sure you have the following prerequisites:</p>
<ol>
<li>Python 3.x installed</li>
<li>NumPy and Matplotlib libraries installed</li>
<li>KITTI dataset downloaded and extracted</li>
<li>OpenCV: for image and video processing</li>
<li>Pykitti: a Python library for working with the KITTI dataset</li>
</ol>

### Install the Required Libraries
<p>Once you have downloaded the dataset, you will need to install the required libraries to work with it in Python. The following libraries are commonly used:</p>

<p>NumPy: for numerical operations and array manipulation
OpenCV: for image and video processing
Matplotlib: for data visualization
You can install these libraries using pip, the Python package manager, by running the following commands in your terminal:</p>

```shell
pip install numpy
pip install opencv-python
pip install matplotli
pip install plotly
pip install glob
pip install progressbar
```


### Load the Dataset
<p>The KITTI Odometry dataset consists of multiple sequences, each containing a set of stereo image pairs and corresponding ground truth poses. We will load the data using the <code class="language-plaintext highlighter-rouge">cv2</code> library in Python.</p>

```python
# Import the libraries
import numpy as np
import cv2
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Define the paths to the data
path_img = "./data_odometry_gray/dataset/sequences/"
path_pose = "./data_odometry_poses/dataset/poses/"

# Load the pose data using pandas
num = "00"
poses = pd.read_csv(path_pose + ("%s.txt" % num), delimiter=' ', header=None)

# Extract the ground truth coordinates from the pose data
gt = np.array([np.array(poses.iloc[i]).reshape((3, 4)) for i in range(len(poses))])

# Extracting spatial coordinates from the gt tensor
z = gt[:, :, 3][:, 2]
y = gt[:, :, 3][:, 1]
x = gt[:, :, 3][:, 0]

# Creating a 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='lines',
    line=dict(color='red', width=2)
)])

# Customize the layout of the plot
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z',
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=0.6),
        projection=dict(type='perspective')
    ),
    bgcolor='whitesmoke',
    xaxis=dict(showgrid=True, gridwidth=0.8, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=0.8, gridcolor='lightgray'),
    zaxis=dict(showgrid=True, gridwidth=0.8, gridcolor='lightgray')
))

# Add a title to the plot for better context
fig.update_layout(title='Spatial Trajectory', title_font=dict(size=16, color='black'))

# Display the spatial trajectory plot
fig.show()
```

<iframe src="/kitti-gt-00.html" width="800" height="600" frameborder="0"></iframe>



```python
# Load and display a test image
test_img = cv2.imread(path_img + num + '/image_0/000000.png')
plt.figure(figsize=(12, 6))
plt.imshow(test_img)
plt.axis('off')
plt.title('Test Image')
plt.show()
```

<img id="myImg" src="/kitti/00.png">

## Understanding Calibration and Timestamp Data in 3D Vision Applications

In 3D vision applications, accurate calibration and synchronized timestamps play a crucial role in aligning and mapping data from multiple sensors. In following, we will explore the contents of two important files: calib.txt and times.txt. These files provide essential information for camera calibration and synchronization of image pairs. Understanding these concepts is vital for accurately transforming and processing data in 3D vision applications.

In computer vision and camera calibration, intrinsic and extrinsic matrices are used to describe the properties and transformations of a camera.

### Intrinsic Matrix
The intrinsic matrix, denoted as K, represents the internal properties of a camera which describes the mapping between the 3D world coordinates and the 2D image plane coordinates of a camera without considering any external factors. It includes parameters such as focal length $(f_x, f_y)$ principal point $(c_x, c_y)$, and skew $(s)$ which define the camera's optical properties, including how the image is formed on the camera sensor. The intrinsic matrix is typically a 3x3 matrix:

<div>
$$
K = \begin{bmatrix}
    f_x & s & c_x \\
    0 & f_y & c_y \\
    0 & 0 & 1 \\
\end{bmatrix}
$$
</div>



### Extrinsic Matrix
The extrinsic matrix represents the external properties or transformations of a camera, specifically its position and orientation in the world which describes the transformation from the camera's coordinate system to a global coordinate system. It includes rotation and translation parameters. The rotation matrix, denoted as R, represents the rotation of the camera, while the translation vector, denoted as t, represents the translation (displacement) of the camera in the global coordinate system. The extrinsic matrix is typically a 3x4 matrix:

$$
\begin{bmatrix}
    R & | &t
\end{bmatrix}
$$

The extrinsic matrix allows us to map points from the camera's coordinate system to the global coordinate system or vice versa. It is used to transform 3D world coordinates into the camera's coordinate system, enabling the projection of these points onto the image plane.

In summary, the intrinsic matrix captures the internal properties of a camera, such as focal length and principal point, while the extrinsic matrix describes the camera's position and orientation in the world. Together, these matrices enable the transformation between 3D world coordinates and 2D image plane coordinates.

### Calibration Data (calib.txt):
The calib.txt file contains calibration data for the cameras used in the system, providing the projection matrices and a transformation matrix necessary for mapping points between different coordinate systems. Let's break down the contents of the calib.txt file:

* Projection Matrices ($P_0$, $P_1$, $P_2$, $P_3$):
The projection matrices, $P_0$, $P_1$, $P_2$, and $P_3$, are $3 \times 4$ matrices that represent the transformation from 3D world coordinates to 2D image plane coordinates for each camera in the system. $P_0$ corresponds to the left camera, $P_1$ corresponds to the right camera, $P_2$ corresponds to the third camera, and $P_3$ corresponds to the fourth camera. These matrices are crucial for rectifying and projecting the 3D point clouds onto the image planes which contain intrinsic information about each camera's focal length and optical center. Moreover, they also include transformation information that relates the coordinate frames of each camera to the global coordinate frame.

The projection matrix could be defined as follows:

$$
\begin{equation}
P = K\begin{bmatrix}
    R  | t
\end{bmatrix}
\end{equation}
$$

The projection matrix enables the projection of 3D coordinates from the global frame onto the image plane of the camera using the following equation:

<div>
$$
\begin{equation}
\lambda \begin{bmatrix}
u \\ v \\ 1
\end{bmatrix} = P\begin{bmatrix}
    X \\ Y \\ Z \\ 1
\end{bmatrix}
\end{equation}
$$
</div>

Here, $\lambda$ represents a scaling factor, $(u, v)$ are the coordinates of the projected point on the image plane, and $(X, Y, Z)$ are the coordinates of the 3D point in the global frame. The projection matrix $P$ encapsulates both the intrinsic and extrinsic parameters of the camera, allowing for the transformation between 3D world coordinates and 2D image plane coordinates.

By decomposing the projection matrix $P$ into intrinsic and extrinsic camera matrices, we can obtain a more explicit description of the process involved in projecting a 3D point from any coordinate frame onto the pixel coordinate frame of the camera. This breakdown allows us to separate the intrinsic parameters, which capture the internal characteristics of the camera such as focal length and optical center, from the extrinsic parameters that define the camera's position and orientation in the global coordinate system. By considering the intrinsic and extrinsic matrices individually, we gain a deeper understanding of how the camera's internal properties and its relationship to the global coordinate system influence the projection process.

<div>
$$
\begin{equation}
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix} =  \frac{1}{\lambda} K \begin{bmatrix}R|t\end{bmatrix} \begin{bmatrix}X \\ Y \\ Z \\ 1 \end{bmatrix}
\end{equation}
$$
</div>

Before examining the provided code, let's set some expectations based on standard projection matrices for each camera. If the projection matrices follow the standard convention, we would anticipate the extrinsic matrix to transform a point from the global coordinate frame into the camera's own coordinate frame. To illustrate this, let's consider the scenario where we take the origin of the global coordinate frame (corresponding to the origin of the left grayscale camera) and translate it into the coordinate frame of the right grayscale camera. In this case, we would expect the X coordinate to be -0.54 since the left camera's origin is positioned 0.54 meters to the left of the right camera's origin. To verify this expectation, we can decompose the projection matrix provided for the right camera into intrinsic matrix (k), rotation matrix (R), and translation vector (t) using the helpful function available in OpenCV. By printing these decomposed matrices and vectors, we can examine their values to determine if they align with our expectations.


```python
# Extracting intrinsic and extrinsic parameters from the projection matrix P1
P1 = np.array(calib.loc['P1:']).reshape((3, 4))
k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
t1 = t1 / t1[3]

# Displaying the results
print('Intrinsic Matrix:')
print(k1)
print('Rotation Matrix:')
print(r1)
print('Translation Vector:')
print(t1.round(4))
```

This code segment extracts the intrinsic and extrinsic parameters from the projection matrix `P1`. The `decomposeProjectionMatrix` function from the OpenCV library is used to decompose the matrix into the intrinsic matrix `k1`, rotation matrix `r1`, translation vector `t1`, and other related information. The translation vector is then normalized by dividing it by its fourth component. The extracted matrices and vector are then displayed, providing insight into the intrinsic properties (intrinsic matrix) of the camera, as well as its orientation (rotation matrix) and position (translation vector) in the 3D space.


```Intrinsic Matrix:
[[718.856    0.     607.1928]
 [  0.     718.856  185.2157]
 [  0.       0.       1.    ]]
Rotation Matrix:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Translation Vector:
[[ 0.5372]
 [ 0.    ]
 [-0.    ]
 [ 1.    ]]

```

Now that we understand the calibration matrices and their reference to the left grayscale camera's image plane, let's revisit the projection equation and discuss the lambda $ (λ)$ value. Lambda represents the depth of a 3D point after applying the transformation $[R|t]$ to the point. It signifies the point's distance from the camera. Since we are projecting onto a 2D plane, dividing each point by its depth effectively projects them onto a plane located one unit away from the camera's origin along the Z-axis, resulting in a Z value of 1 for all points. It is important to note that the division by lambda can be performed at any point during the operations without affecting the final result. We will demonstrate this concept by performing the computations in Python.

```python
some_point = np.array([1, 2, 3, 1]).reshape(-1, 1)
transformed_point = gt[14].dot(some_point)
depth_from_cam = transformed_point[2]

print('Original point:\n', some_point)
print('Transformed point:\n', transformed_point.round(4))
print('Depth from camera:\n', depth_from_cam.round(4))
```

In this code, `some_point` represents a point measured in the coordinate frame of the left camera at its 14th pose. We then transform this point to the camera's coordinate frame using the transformation matrix `gt[14]` and extract the Z coordinate to obtain the depth from the camera. The code prints the original point, the transformed point, and the depth from the camera.

```
Original point:
 [[1]
 [2]
 [3]
 [1]]
Transformed point:
 [[ 0.2706]
 [ 1.5461]
 [15.0755]]
Depth from camera:
 [15.0755]
 ```

 To project a 3D point onto the image plane, we have two approaches: either applying the intrinsic matrix after dividing by the depth or dividing by the depth first and then multiplying by the intrinsic matrix. Here's the code to demonstrate both ways:

```python
# Multiplying by intrinsic matrix k, then dividing by depth
pixel_coordinates1 = k1.dot(transformed_point) / depth_from_cam

# Dividing by depth then multiplying by intrinsic matrix k
pixel_coordinates2 = k1.dot(transformed_point / depth_from_cam)

print('Pixel Coordinates (Approach 1):', pixel_coordinates1.T)
print('Pixel Coordinates (Approach 2):', pixel_coordinates2.T)
```

In this code, `pixel_coordinates1` represents the pixel coordinates obtained by first multiplying the transformed point by the intrinsic matrix `k1` and then dividing by the depth. Similarly, `pixel_coordinates2` represents the pixel coordinates obtained by dividing the transformed point by the depth and then multiplying by `k1`. The code prints the pixel coordinates for both approaches.

```
Pixel Coordinates (Approach 1): [[620.09802465 258.93763336   1.        ]]
Pixel Coordinates (Approach 2): [[620.09802465 258.93763336   1.        ]]
```

* Transformation Matrix $(T_r)$:
The transformation matrix, $T_r$, performs the conversion from the Velodyne scanner's coordinate system to the left rectified camera's coordinate system. This transformation is necessary to map a point from the Velodyne scanner to the corresponding point in the left image plane.

* Mapping a Point:
To map a point $X$ from the Velodyne scanner to a point $x$ in the $i_{th}$ image plane, you need to perform the following transformation:

$$
\begin{equation}
x = P_i \cdot Tr \cdot X
\end{equation}
$$

Here, $P_i$ represents the projection matrix of the $i_{th}$ camera, and $X$ is the point in the Velodyne scanner's coordinate system.

### Timestamp Data (times.txt):
The times.txt file provides timestamps for synchronized image pairs in the KITTI dataset. These timestamps are expressed in seconds and are crucial for analyzing the dynamics of the vehicle and understanding the timing of image acquisition. By considering the timing information, you can accurately associate images from different cameras, Lidar scans, or other sensors that are synchronized with the image acquisition system.

To access and utilize the timestamps in times.txt, you can read the file and load it into a suitable data structure in Python. Here's an example using the pandas library:

```python
import pandas as pd

# Read the times.txt file
times = pd.read_csv(path_img + num +'/times.txt', delimiter=' ', header=None)

# Display the first few rows
times.head()
```

By loading the timestamps into a dataframe, you can effectively analyze the relationships between different sensor data and synchronize them based on the timestamps. This allows for accurate alignment and fusion of data from multiple sensors, enabling more robust and comprehensive analysis in 3D vision applications.


## Processing Image Data with Timestamps

Once you have the synchronized timestamps from the times.txt file, you can use them to associate images from different cameras or perform temporal analysis of the data. Here's an example of how you can process image data with timestamps in Python:

```python
import cv2
import os
import pandas as pd

# Read the times.txt file
times = pd.read_csv(path_img + num + '/times.txt', delimiter=' ', header=None)

# Define the path to the image directory
image_dir = path_img + num + '/image_0/'

# Get a list of image files in the directory
image_files = sorted(os.listdir(image_dir))

# Iterate over the image files and process them
for i, image_file in enumerate(image_files):
    # Construct the image file path
    image_path = os.path.join(image_dir, image_file)

    # Load and process the image
    image = cv2.imread(image_path)

    # Get the corresponding timestep
    timestep = times.iloc[i][0]

    # Draw the timestep on the image
    cv2.putText(image, f'Timestep: {timestep}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Perform image processing tasks
    # ...

    # Display the image or perform further analysis
    cv2.imshow('Image', image)

    # Wait for a key press (maximum delay of 10 milliseconds)
    key = cv2.waitKey(0)

    # Break the loop if the 'ESC' key is pressed (ASCII code 27)
    if i >= 25:
        break
        
# Release any open windows
cv2.destroyAllWindows()

```

In this example, we first read the timestamps from the times.txt file using pandas. Then, we iterate over the timestamps and process the corresponding images. Inside the loop, we format the timestamp and construct the image filename based on the timestamp. We load the image using OpenCV's `cv2.imread()` function and perform any desired image processing tasks. You can apply various computer vision techniques, such as object detection, image segmentation, or feature extraction, to analyze the images.

After processing the image, you can display it using OpenCV's `cv2.imshow()` function or perform further analysis based on your requirements. In this example, we break the loop after processing a few images (for demonstration purposes) by checking the iteration count. However, you can remove this condition to process all the images.

Finally, we release any open windows using `cv2.destroyAllWindows()` to clean up after processing all the images.

By using the synchronized timestamps, you can ensure that the images from different cameras or sensors are processed and analyzed in the correct temporal order, enabling more accurate and meaningful results in your 3D vision applications.


Below is an example of a dataset handling object that can help in accessing and processing the data, while also explicitly decoding the Velodyne binaries as float32:


```python
class Dataset_Handler():
    def __init__(self, sequence, lidar=True, progress_bar=True, low_memory=True):
        import pandas as pd
        import os
        import cv2
        import numpy as np
        from tqdm import tqdm

        self.lidar = lidar
        self.low_memory = low_memory

        self.seq_dir = './data_odometry_gray/dataset/sequences/{}/'.format(sequence)
        self.poses_dir = './data_odometry_poses/dataset/poses/{}.txt'.format(sequence)
        poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)

        self.left_image_files = os.listdir(self.seq_dir + 'image_0')
        self.right_image_files = os.listdir(self.seq_dir + 'image_1')
        self.velodyne_files = os.listdir(self.seq_dir + 'velodyne')
        self.num_frames = len(self.left_image_files)
        self.lidar_path = self.seq_dir + 'velodyne/'

        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3,4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3,4))
        self.P2 = np.array(calib.loc['P2:']).reshape((3,4))
        self.P3 = np.array(calib.loc['P3:']).reshape((3,4))
        self.Tr = np.array(calib.loc['Tr:']).reshape((3,4))

        self.times = np.array(pd.read_csv(self.seq_dir + 'times.txt', delimiter=' ', header=None))
        self.gt = np.zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

        if self.low_memory:
            self.reset_frames()
            self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[0], 0)
            self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' + self.right_image_files[0], 0)
            self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1], 0)
            if self.lidar:
                self.first_pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[0],
                                                    dtype=np.float32,
                                                    count=-1).reshape((-1, 4))
            self.imheight = self.first_image_left.shape[0]
            self.imwidth = self.first_image_left.shape[1]

        if progress_bar:
            bar = tqdm(total=self.num_frames, desc='Loading Images')

        self.images_left = []
        self.images_right = []
        self.pointclouds = []

        for i, name_left in enumerate(self.left_image_files):
            name_right = self.right_image_files[i]
            self.images_left.append(cv2.imread(self.seq_dir + 'image_0/' + name_left))
            self.images_right.append(cv2.imread(self.seq_dir + 'image_1/' + name_right))
            if self.lidar:
                pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[i],
                                         dtype=np.float32,
                                         count=-1).reshape([-1, 4])
                self.pointclouds.append(pointcloud)
            if progress_bar:
                bar.update(1)

        if progress_bar:
            bar.close()

        self.imheight = self.images_left[0].shape[0]
        self.imwidth = self.images_left[0].shape[1]
        self.first_image_left = self.images_left[0]
        self.first_image_right = self.images_right[0]
        self.second_image_left = self.images_left[1]
        if self.lidar:
            self.first_pointcloud = self.pointclouds[0]
        else:
            if progress_bar:
                bar = tqdm(total=self.num_frames, desc='Loading Images')

            self.images_left = []
            self.images_right = []
            self.pointclouds = []

            for i, name_left in enumerate(self.left_image_files):
                name_right = self.right_image_files[i]
                self.images_left.append(cv2.imread(self.seq_dir + 'image_0/' + name_left))
                self.images_right.append(cv2.imread(self.seq_dir + 'image_1/' + name_right))
                if self.lidar:
                    pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[i],
                                             dtype=np.float32,
                                             count=-1).reshape([-1,4])
                    self.pointclouds.append(pointcloud)
                if progress_bar:
                    bar.update(1)

            if progress_bar:
                bar.close()

            self.imheight = self.images_left[0].shape[0]
            self.imwidth = self.images_left[0].shape[1]
            self.first_image_left = self.images_left[0]
            self.first_image_right = self.images_right[0]
            self.second_image_left = self.images_left[1]
            if self.lidar:
                self.first_pointcloud = self.pointclouds[0]

    def reset_frames(self):
        self.images_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0)
                            for name_right in self.right_image_files)
        if self.lidar:
            self.pointclouds = (np.fromfile(self.lidar_path + velodyne_file,
                                            dtype=np.float32,
                                            count=-1).reshape((-1, 4))
                                for velodyne_file in self.velodyne_files)
```

To run that, you can use the code below:

```python
handler = Dataset_Handler('04')
```

We used the sequence `04` as its much smaller than other sequences which makes it proper for the excercise purposes.



## Depth Maps and Visual Odometry


## Calculating Baseline and Focal Length in Stereo Vision

Stereo vision involves using a pair of cameras to capture images of a scene from slightly different viewpoints. By analyzing the disparities between corresponding points in the stereo images, we can estimate depth information and reconstruct the 3D structure of the scene. To perform accurate depth estimation, it is essential to know the baseline (distance between the cameras) and the focal length (effective focal length of the cameras). Let's explore how to calculate these parameters:

### Baseline Calculation

The baseline is the distance between the two cameras in the stereo rigw hich determines the scale of the depth information captured by the cameras. To calculate the baseline, we need to determine the translation between the camera coordinate frames.

One way to obtain the translation is by decomposing the projection matrix of one of the cameras using OpenCV's `decomposeProjectionMatrix` function, which can extract the translation vector $t$ from the projection matrix $P$. The baseline is then given by the absolute value of the x-component of the translation vector:

$$
\begin{equation}
    \text{baseline} = \text{abs}(t[0])
\end{equation}
$$

### Focal Length Calculation

The focal length represents the effective focal length of the cameras used in the stereo rig. It determines the scale at which the scene is captured. To calculate the focal length, we need to extract the intrinsic parameters from the camera calibration matrix.

The camera calibration matrix, denoted as $K$, contains the focal length and the principal point coordinates. By accessing the appropriate elements of the calibration matrix, we can obtain the focal length:

$$
\begin{equation}
    \text{focal length} = K[0, 0]
\end{equation}
$$


By calculating the baseline and focal length, we can accurately estimate depth information using stereo vision. These parameters are crucial for various applications, such as 3D reconstruction, object detection, and visual odometry.

Remember that accurate calibration of the stereo rig is essential for reliable depth estimation. Calibration involves accurately determining the intrinsic and extrinsic parameters of the cameras. Calibration techniques such as chessboard calibration or pattern-based calibration can be used to obtain precise calibration parameters.

Here's an example Python code snippet that demonstrates how to calculate the baseline and focal length using the provided projection matrix:


```python
import numpy as np
import cv2

# Projection matrix for the right camera
P1 = np.array(calib.loc['P1:']).reshape((3, 4))

# Decompose the projection matrix
k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
t1 = t1 / t1[3]

# Calculate the baseline (distance between the cameras)
baseline = abs(t1[0][0])

# Calculate the focal point (effective focal length of the cameras)
focal_length = k1[0][0]

# Print the results
print('Baseline:', baseline, 'meters')
print('Focal Length:', focal_length, 'pixels')
```

```
For the seq "00":
Baseline: 0.5371657188644179 meters
Focal Length: 718.8559999999999 pixels
```

## Calculating Depth from Stereo Pair of Images

In stereo vision, depth can be estimated by calculating the disparity between corresponding pixels in a stereo pair of images. The disparity represents the horizontal shift of a pixel between the left and right images and is inversely proportional to the depth of the corresponding 3D point. Let's delve into the technical details and mathematical equations involved in this process:

### Rectification

Before calculating disparity, it is crucial to rectify the stereo pair of images to ensure that corresponding epipolar lines are aligned. Rectification simplifies the correspondence matching process. It involves finding the epipolar geometry, computing rectification transformations, and warping the images accordingly.

The rectification process transforms the left and right images, denoted as $I_l$ and $I_r$ respectively, into rectified images $I_l'$ and $I_r'$. This transformation ensures that corresponding epipolar lines in the rectified images are aligned horizontally. The rectification transformations consist of rotation matrices $R_l$ and $R_r$ and translation vectors $T_l$ and $T_r$.

Given a point $(x_l, y_l)$ in the left image, its corresponding point $(x_r, y_r)$ in the right image can be obtained using the rectification transformations as follows:

<div>
$$
\begin{equation}
\begin{pmatrix} x_r \\ y_r \\ 1 \end{pmatrix} = R_r \cdot R_l^T \cdot \begin{pmatrix} x_l \\ y_l \\ 1 \end{pmatrix} + T_r - R_r \cdot T_l
\end{equation}
$$
</div>

After rectification, the corresponding epipolar lines in the rectified images are aligned horizontally, simplifying the correspondence matching process.

### Correspondence Matching

Once the stereo images are rectified, the next step is to find corresponding points between the two images. This process is known as correspondence matching or stereo matching. The goal is to find the best matching pixel or pixel neighborhood in the right image for each pixel in the left image.

Various algorithms can be used for correspondence matching, such as block matching, semi-global matching, or graph cuts. These algorithms search for the best matching pixel or pixel neighborhood in the other image based on similarity measures like sum of absolute differences (SAD) or normalized cross-correlation (NCC).

The correspondence matching process involves searching for the disparity value that minimizes the dissimilarity measure between the left and right image patches. Let $P_l$ denote the pixel patch centered at $(x_l, y_l)$ in the left image, and $P_r$ denote the corresponding pixel patch centered at $(x_r, y_r)$ in the right image. The disparity $d$ for the pixel pair $(x_l, y_l)$ and $(x_r, y_r)$ can be computed as:

$$
\begin{equation}
d = x_l - x_r
\end{equation}
$$

### Disparity Computation

The disparity value represents the shift or offset between the corresponding points in the stereo pair of images. It is computed as the horizontal distance between the pixel coordinates in the rectified images. The disparity can be calculated using the formula:

$$
\begin{equation}
\text{{Disparity}} = x_l - x_r
\end{equation}
$$

where $x_l$ is the x-coordinate of the pixel in the left image, and $x_r$ is the x-coordinate of the corresponding pixel in the right image.

### Depth Estimation

Once the disparity is computed, the depth can be estimated using the disparity-depth relationship. This relationship is based on the geometry of the stereo camera setup and assumes a known baseline (distance between the left and right camera) and focal length.

Let $B$ denote the baseline (distance between the left and right camera), and $f$ denote the focal length (effective focal length of the cameras). The depth $Z$ can be calculated using the formula:

$$
\begin{equation}
Z = \frac{{B \cdot f}}{{\text{{Disparity}}}}
\end{equation}
$$

where the disparity represents the computed disparity value for the pixel pair.

The resulting depth map provides the depth information for each pixel in the rectified image, representing the 3D structure of the scene.

### Depth Refinement

The computed depth values may contain noise and outliers. To improve the accuracy and smoothness of the depth map, post-processing techniques can be applied.

One common technique is depth map filtering, where a filter is applied to each pixel or a neighborhood of pixels to refine the depth values. The filter can be based on techniques like weighted median filtering, bilateral filtering, or joint bilateral filtering.

Another approach is depth map inpainting, which fills in missing or erroneous depth values based on the surrounding valid depth values. Inpainting algorithms utilize neighboring information to estimate missing or unreliable depth values.

Guided filtering is another technique that can be used to refine the depth map. It uses a guidance image, such as the rectified left image, to guide the filtering process and preserve edges and details in the depth map.

By applying these post-processing techniques, the accuracy and quality of the depth map can be improved.

By following these steps and applying the corresponding formulas and equations, accurate depth maps can be generated from stereo pairs of images. These depth maps provide valuable information for various applications, including 3D reconstruction, scene understanding, and autonomous navigation.

Remember that the accuracy of depth estimation depends on factors such as image quality, camera calibration, and the chosen correspondence matching algorithm. Experimentation and fine-tuning of parameters may be necessary to achieve optimal results.

Apologies for the oversight. Here's the section you mentioned:

## Math behind Stereo Depth

To review the essentials of the math behind stereo depth,
Using similar triangles, we can write the following equations:

$$
\begin{equation}
\frac{{X}}{{Z}} = \frac{{x_L}}{{f}}
\end{equation}
$$

$$
\begin{equation}
\frac{{X - B}}{{Z}} = \frac{{x_R}}{{f}}
\end{equation}
$$

Where $X$ represents the 3D x-coordinate of the point, $Z$ represents the depth of the point, $x_L$ and $x_R$ represent the x-coordinates of the point in the left and right image planes respectively, $f$ represents the focal length, and $B$ represents the baseline (distance between the cameras).

We define disparity, $d$, as the difference between $x_L$ and $x_R$, which represents the difference in horizontal pixel location of the point projected onto the left and right image planes:

$$
\begin{equation}
d = x_L - x_R
\end{equation}
$$

Rearranging the similar triangles equations, we obtain:

$$
\begin{equation}
\frac{{X}}{{Z}} = \frac{{x_L}}{{f}} 
\end{equation}
$$

$$
\begin{equation}
\frac{{X - B}}{{Z}} = \frac{{x_R}}{{f}} 
\end{equation}
$$

By substituting $Z \cdot x_L$ into $f \cdot X$ in Equation 2, we get:

$$
\begin{equation}
\frac{{Z \cdot x_L - B \cdot x_L}}{{Z}} = \frac{{x_R}}{{f}}
\end{equation}
$$

This equation can be rewritten as:

$$
\begin{equation}
x_L - \frac{{B \cdot x_L}}{{Z}} = x_R 
\end{equation}
$$

Next, we substitute the definition of disparity ($d = x_L - x_R$) into Equation 3 and solve for $Z$:

$$
\begin{equation}
x_L - \frac{{B \cdot x_L}}{{Z}} = x_L - d
\end{equation}
$$

$$
\begin{equation}
\frac{{B \cdot x_L}}{{Z}} = d
\end{equation}
$$

$$
\begin{equation}
\frac{{B}}{{Z}} = \frac{{d}}{{x_L}} 
\end{equation}
$$

Finally, by rearranging Equation 4, we can solve for $Z$:

$$
\begin{equation}
Z = \frac{{B \cdot f}}{{d}}
\end{equation}
$$

Note that if the focal length and disparity are measured in pixels, the pixel units will cancel out, and if the baseline is measured in meters, the resulting $Z$ measurement will be in meters, which is desirable for reconstructing 3D coordinates later.

With the baseline (previously found to be 0.54m), the focal length of the x-direction in pixels (previously found to be 718.856px), and the disparity between the same points in the two images, we now have a simple way to compute the depth from our stereo pair of images.

To find the necessary disparity value, we can use the stereo matching algorithms available in OpenCV, such as StereoBM or StereoSGBM. StereoBM is faster, while StereoSGBM produces better results, as we will see in practice.

By applying these equations and utilizing the appropriate stereo matching algorithm, we can estimate the depth accurately and reconstruct 3D coordinates from the stereo pair of images.


At first we define two vital parameters, "SAD window" and "block size". In the context of stereo vision and disparity mapping, the terms "SAD window" and "block size" refer to parameters that affect the matching algorithm used to calculate disparities between corresponding points in the left and right stereo images.

1. SAD Window (Sum of Absolute Differences Window):
The SAD window, also known as the matching window or kernel, defines the neighborhood around a pixel in the left image that is considered for matching with the corresponding neighborhood in the right image. It represents the region within which the algorithm searches for similarities to determine the disparity.

The size of the SAD window determines the spatial extent of the matching process. A larger window size allows for capturing more context and potentially leads to more accurate disparity estimation, but it also increases computational complexity. The SAD window is usually a square or rectangular region centered around each pixel.

2. Block Size:
The block size specifies the dimensions of the blocks or patches within the SAD window that are compared during the matching process. It defines the size of the local regions used for computing disparities.

A larger block size means that more pixels are considered during matching, which can improve robustness against noise and texture variations. However, increasing the block size also increases the computational cost. Typically, the block size is an odd number to have a well-defined center pixel for comparison.

Both the SAD window size and block size are important parameters that influence the trade-off between accuracy and computational efficiency in stereo matching algorithms. The optimal values for these parameters depend on factors such as image resolution, scene complexity, and noise characteristics, and they may require experimentation and tuning for specific applications.

To compute the disparity map for the left image using the specified matcher we use the function below:

```python
def compute_disp_left(img_left, img_right,sad_window,block_size, matcher_name='bm'):
    '''
    Arguments:
    img_left -- image from the left camera
    img_right -- image from the right camera
    
    Optional Argument:
    matcher_name -- (str) the matcher name, can be 'bm' for StereoBM or 'sgbm' for StereoSGBM matching
    
    Returns:
    disp_left -- disparity map for the left camera image
    '''
    
    # Set the parameters for disparity calculation
    num_disparities = sad_window * 16    
    # Convert the input images to grayscale
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Create the stereo matcher based on the selected method
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    elif matcher_name == 'sgbm':                             
        matcher = cv2.StereoSGBM_create(
            numDisparities=num_disparities,
            minDisparity=0,
            blockSize=block_size,
            P1=8 * 3 * sad_window ** 2,
            P2=32 * 3 * sad_window ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    # Compute the disparity map
    disparity = matcher.compute(img_left_gray, img_right_gray)

    # Normalize the disparity map for visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Compute the disparity map for the left camera image and return it
    disp_left = disparity.astype(np.float32) / 16
    return disp_left
```

```python
sad_window = 5
block_size = 11

disp_left = compute_disp_left(handler.first_image_left, handler.first_image_right, sad_window, block_size, matcher_name='bm')
# Set the figure size
plt.figure(figsize=(11, 7))
# Set the plot title
plt.title('Disparity Map (Left Image) - Stereo Block Matching')
# Display the disparity map
plt.imshow(disp_left)
# Add axis labels
plt.xlabel('Pixel Columns')
plt.ylabel('Pixel Rows')
# Show the plot
plt.show()

# Compute the disparity map for the left image
disp_left = compute_disp_left(handler.first_image_left, handler.first_image_right, sad_window, block_size, matcher_name='sgbm')
# Set the figure size
plt.figure(figsize=(11, 7))
# Set the plot title
plt.title('Disparity Map (Left Image) - Stereo Semi-Global Block Matching')
# Display the disparity map
plt.imshow(disp_left)
# Add axis labels
plt.xlabel('Pixel Columns')
plt.ylabel('Pixel Rows')
# Show the plot
plt.show()
```


<img id="myImg" src="/kitti/kitti-StereoBM.png">
<img id="myImg" src="/kitti/kitti-StereoSGBM.png">


### StereoBM Vs. StereoSGBM

StereoBM and StereoSGBM are algorithms used in stereo vision for computing the disparity map from a pair of stereo images which are both provided by the OpenCV library and offer different approaches to stereo matching.

1. StereoBM (Stereo Block Matching):
   - StereoBM is a block matching algorithm that operates on grayscale images.
   - It works by comparing blocks of pixels between the left and right images and finding the best matching block based on a predefined matching cost.
   - The disparity is then computed by calculating the difference in the horizontal pixel coordinates of the matching blocks.
   - StereoBM is relatively fast and suitable for real-time applications but may not handle challenging scenes with textureless or occluded regions as effectively.

2. StereoSGBM (Stereo Semi-Global Block Matching):
   - StereoSGBM is an extension of the StereoBM algorithm that addresses some of its limitations.
   - It incorporates additional steps such as semi-global optimization to refine the disparity map and handle occlusions and textureless regions more robustly.
   - StereoSGBM also considers a wider range of disparities during the matching process, allowing for more accurate depth estimation.
   - It takes into account various cost factors, such as the uniqueness of matches and the smoothness of disparities, to improve the overall quality of the disparity map.
   - However, StereoSGBM is computationally more expensive than StereoBM due to the additional optimization steps involved.

Both StereoBM and StereoSGBM are widely used in stereo vision applications for tasks such as 3D reconstruction, depth estimation, object detection, and robot navigation. The choice between the two algorithms depends on the specific requirements of the application, including the scene complexity, computational resources available, and desired accuracy.


The code below is used to visualize the disparity map obtained from stereo image pairs. The disparity map contains depth information about the scene, enabling us to perceive the 3D structure of objects within the images. By overlaying the disparity map onto the main image, we can visualize the depth variations and understand the relative distances of objects in the scene. This visualization aids in applications such as depth estimation, 3D reconstruction, object detection, and scene understanding.

```python

# Compute the disparity map
disp_left = compute_disp_left(handler.first_image_left, handler.first_image_right, sad_window, block_size, matcher_name='bm')

# Display the main image with the disparity map as overlay
plt.figure(figsize=(11, 7))
plt.title('Disparity Map Overlay: Visualizing Depth Information in Stereo Images (BM)')

plt.imshow(handler.first_image_left)
plt.imshow(disp_left, alpha=0.6, cmap='jet')
# Add axis labels
plt.xlabel('Pixel Columns')
plt.ylabel('Pixel Rows')
# Show the plot
plt.show()



# Compute the disparity map
disp_left = compute_disp_left(handler.first_image_left, handler.first_image_right, sad_window, block_size, matcher_name='sgbm')

# Display the main image with the disparity map as overlay
plt.figure(figsize=(11, 7))
plt.title('Disparity Map Overlay: Visualizing Depth Information in Stereo Images (SGBM)')

plt.imshow(handler.first_image_left)
plt.imshow(disp_left, alpha=0.6, cmap='jet')
# Add axis labels
plt.xlabel('Pixel Columns')
plt.ylabel('Pixel Rows')
# Show the plot
plt.show()

```


<img id="myImg" src="/kitti/real_depth_img_bm.png">
<img id="myImg" src="/kitti/real_depth_img_sgbm.png">


## Feature Extraction

Feature extraction plays a critical role in computer vision tasks by capturing and representing distinct patterns or characteristics of images or image regions. These features serve as meaningful representations that enable higher-level analysis and interpretation of visual data. In this section, we will explore some advanced techniques for feature extraction in computer vision.


### 1. Harris Corner Detection

Harris Corner Detection, proposed by  [<a id="Harris" href="#HarrisRef">Harris and Stephens</a>] in 1988. It is a widely used algorithm for detecting corners in images. Corner points represent the junctions of two or more edges, which are highly informative for image matching and tracking. The Harris Corner Detection algorithm consists of the following steps:

1. **Image Gradient Calculation:** Compute the gradients of the image in the x and y directions using techniques such as Sobel or Prewitt operators.

2. **Structure Tensor Computation:** Based on the gradients, construct the structure tensor for each pixel. The structure tensor is a matrix that represents the local image structure at a given point.

3. **Corner Response Function:** Compute the corner response function using the structure tensor. The corner response function measures the likelihood of a pixel being a corner based on the eigenvalues of the structure tensor.

4. **Non-maximum Suppression:** Apply non-maximum suppression to the corner response function to select the most prominent corners while suppressing weaker nearby corners.

The Harris Corner Detection algorithm can be described using the following equations:

- Gradient calculation:
  $$ \begin{equation} I_x = \frac{{\partial I}}{{\partial x}} \quad \text{and} \quad I_y = \frac{{\partial I}}{{\partial y}} \end{equation} $$

- Structure tensor computation:
  $$ \begin{equation} M = \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix} \end{equation} $$

- Corner response function:
  $$ \begin{equation} R = \text{det}(M) - k \cdot \text{trace}(M)^2 \end{equation} $$

- Non-maximum suppression:
  - Select local maxima of R by comparing R with a threshold and considering neighboring pixels.

where, $I_x$  and $I_y$ are image derivatives in x and y directions respectively, $det(M)=λ1λ2$ $\text{trace}(M)=λ1+λ2$, $λ1$ and $λ2$ are the eigenvalues of $M$

Harris Corner Detection is a fundamental technique in many computer vision applications, including feature matching, image alignment, and camera calibration. The detected corners serve as robust landmarks that can be used for subsequent analysis and understanding of image content.

```python
img = handler.first_image_left
```

```python
# Load the input image
Image = handler.first_image_left
# Harris Corner Detection
gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate the result for marking the corners
dst = cv2.dilate(dst, None)

# Create a copy of the image to display the detected corners
image = Image.copy()

# Threshold for an optimal value, it may vary depending on the image.
image[dst>0.01*dst.max()]=[0,0,255]

# Display the image with detected corners
plt.figure(figsize=(11, 6))
plt.imshow(image, cmap='gray')
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()
```
<img id="myImg" src="/kitti/Harris.png">

### 2. Shi-Tomasi Corner Detector & Good Features to Track

The Shi-Tomasi Corner Detector, also known as the [<a id="Shi" href="#ShiRef">Good Features to Track</a>] algorithm, is an improvement over the Harris Corner Detection algorithm. It provides a more robust and accurate method for detecting corners in an image. This algorithm selects the most distinctive corners based on a corner response measure. The steps involved in the Shi-Tomasi Corner Detection algorithm are as follows:

1. **Image Gradient Calculation:** Compute the gradients of the image in the x and y directions using techniques such as Sobel or Prewitt operators.

2. **Structure Tensor Computation:** Based on the gradients, construct the structure tensor for each pixel.

3. **Eigenvalue Calculation:** Compute the eigenvalues of the structure tensor for each pixel. The eigenvalues represent the principal curvatures of the local image structure.

4. **Corner Response Calculation:** Calculate the corner response measure using the eigenvalues. The corner response measure is defined as the minimum eigenvalue or a combination of the eigenvalues.

5. **Non-maximum Suppression:** Apply non-maximum suppression to select the most significant corners while suppressing weaker nearby corners.

The Shi-Tomasi Corner Detector algorithm can be described using the following equations:

- Gradient calculation:
  $$ \begin{equation} I_x = \frac{{\partial I}}{{\partial x}} \quad \text{and} \quad I_y = \frac{{\partial I}}{{\partial y}} \end{equation} $$

- Structure tensor computation:
  $$ \begin{equation} M = \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix} \end{equation} $$

- Eigenvalue calculation:
  $$ \begin{equation} \lambda_1, \lambda_2 = \text{eigenvalues}(M) \end{equation} $$

- Corner response calculation:
  $$ \begin{equation} R = \min(\lambda_1, \lambda_2) \end{equation} $$

- Non-maximum suppression:
  - Select local maxima of R by comparing R with a threshold and considering neighboring pixels.

The Shi-Tomasi Corner Detector algorithm is widely used in feature tracking, motion estimation, and image registration tasks. It provides more accurate and reliable corner detection compared to the Harris Corner Detector.

Here's an example code snippet for implementing the Shi-Tomasi Corner Detector using OpenCV:

```python
import cv2
import numpy as np

# Load the input image
Image = handler.first_image_left

# Create a copy of the image to display the detected corners
image = Image.copy()

# Set the maximum number of corners to detect
max_corners = 100

# Set the quality level for corner detection
quality_level = 0.01

# Set the minimum Euclidean distance between corners
min_distance = 10

# Apply Shi-Tomasi Corner Detector
corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)

# Convert corners to integer coordinates
corners = np.int0(corners)

# Draw detected corners on the image
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# Display the image with detected corners using matplotlib
plt.figure(figsize=(11, 6))
plt.imshow(image, cmap='gray')
plt.title('Shi-Tomasi Corner Detection')
plt.axis('off')
plt.show()
```

In this code, we use the `cv2.goodFeaturesToTrack()` function in OpenCV to apply the Shi-Tomasi Corner Detector. It takes the image, maximum number of corners, quality level, and minimum distance as parameters. The function returns the detected corners as a NumPy array. We then convert the corners to integers and draw them on the image using circles. Finally, we display the image with the detected corners.

<img id="myImg" src="/kitti/Shi-Tomasi.png">

### 1. Scale-Invariant Feature Transform (SIFT)

The Scale-Invariant Feature Transform (SIFT) algorithm, introduced by Lowe in 1999, is widely used for robust feature extraction. SIFT extracts distinctive features invariant to scale, rotation, and affine transformations. It consists of the following key steps:

1. **Scale-space Extrema Detection:** SIFT applies a Difference of Gaussian (DoG) algorithm to detect potential keypoints in different scales and locations. The DoG is obtained by subtracting blurred versions of an image at multiple scales.

2. **Keypoint Localization:** Keypoints are refined by eliminating low-contrast and poorly localized points based on the local extrema in the DoG scale space.

3. **Orientation Assignment:** Each keypoint is assigned a dominant orientation based on local image gradients. This orientation provides invariance to image rotation.

4. **Descriptor Generation:** SIFT computes a descriptor for each keypoint by considering the local image gradients and orientations. The descriptor captures the distinctive local image properties.

The SIFT algorithm is characterized by the following equations:

- Difference of Gaussian (DoG):
  $$  \begin{equation} D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) \ast I(x, y)  \end{equation} $$

- Gaussian function:
  $$ \begin{equation}  G(x, y, \sigma) = \frac{1}{{2\pi\sigma^2}} \exp\left(-\frac{{x^2 + y^2}}{{2\sigma^2}}\right)  \end{equation} $$

- Keypoint orientation assignment:
  $$  \begin{equation} \theta = \text{atan2}(M_y, M_x)  \end{equation} $$

- Descriptor generation:
  - Divide the region around the keypoint into sub-regions.
  - Compute gradient magnitude and orientation for each pixel in each sub-region.
  - Construct a histogram of gradient orientations in each sub-region.
  - Concatenate the histograms to form the final descriptor.

### 2. Speeded-Up Robust Features (SURF)

The Speeded-Up Robust Features (SURF) algorithm, introduced by Bay et al. in 2006, provides an efficient alternative to SIFT. SURF extracts robust and distinctive features while achieving faster computation. The main steps of SURF are as follows:

1. **Scale-space Extrema Detection:** SURF applies a Hessian matrix-based approach to detect keypoints at multiple scales. The determinant of the Hessian matrix is used to identify potential interest points.

2. **Orientation Assignment:** Similar to SIFT, SURF assigns a dominant orientation to each keypoint using the sum of Haar wavelet responses.

3. **Descriptor Generation:** SURF computes a descriptor based on the distribution of Haar wavelet responses in the neighborhood of each keypoint. The wavelet responses capture the local image properties.

The SURF algorithm involves the following equations:

- Hessian matrix:
  $$ \begin{equation}
  H(x, y, \sigma) = \begin{bmatrix} L_{xx}(x, y, \sigma) & L_{xy}(x, y, \sigma) \\ L_{xy}(x, y, \sigma) & L_{yy}(x, y, \sigma) \end{bmatrix} \end{equation}  $$

- Determinant of the Hessian matrix:
  $$ \begin{equation}\text{Det}(H(x, y, \sigma)) = L_{xx}(x, y, \sigma) \cdot L_{yy}(x, y, \sigma) - (L_{xy}(x, y, \sigma))^2  \end{equation} $$

- Keypoint orientation assignment:
  $$ \begin{equation}\theta = \text{atan2}\left(\sum w \cdot M_y, \sum w \cdot M_x\right) \end{equation} $$

- Descriptor generation:
  - Divide the region around the keypoint into sub-regions.
  - Compute Haar wavelet responses in each sub-region.
  - Construct a descriptor by combining and normalizing the wavelet responses.

These advanced feature extraction techniques, SIFT and SURF, offer powerful capabilities for identifying distinctive image features invariant to various transformations. They form the foundation for many computer vision applications, including object recognition, image stitching, and 3D reconstruction.

## References

<ul>
  <li>Geiger, Andreas, Philip Lenz, and Raquel Urtasun. “Are we ready for autonomous driving? The KITTI vision benchmark suite.” Conference on Computer Vision and Pattern Recognition (CVPR), 2012.</li>
  <li>KITTI Vision Benchmark Suite. http://www.cvlibs.net/datasets/kitti/</li>
  <li><a id="HarrisRef" href="#Harris">Harris, Chris, and Mike Stephens. "A combined corner and edge detector." Alvey vision conference. Vol. 15. No. 50. 1988.</a></li>
  <li><a id="ShiRef" href="#Shi">Shi, Jianbo. "Good features to track." 1994 Proceedings of IEEE conference on computer vision and pattern recognition. IEEE, 1994.</a></li>
</ul>