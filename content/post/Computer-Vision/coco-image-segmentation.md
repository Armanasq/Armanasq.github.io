---
title: "Image Segmentation"
date: 2023-07-02
description: "Introduction"
url: "/computer-vision/tutorial-01/"
showToc: true
math: true
disableAnchoredHeadings: False
tags:
  - ROS
  - Tutorial
---
[&lArr; Computer Vision](/computer-vision/)
<img src="/coco/coco.png" alt="ROS" style="width:100%;display: block;
  margin-left: auto;
  margin-right: auto; margin-top:0px auto" >
        </div>

# Image Segmentation Tutorial using COCO Dataset and Deep Learning


- [Image Segmentation Tutorial using COCO Dataset and Deep Learning](#image-segmentation-tutorial-using-coco-dataset-and-deep-learning)
  - [COCO Dataset Overview](#coco-dataset-overview)
    - [1. Large-Scale Image Collection](#1-large-scale-image-collection)
    - [2. Object Categories](#2-object-categories)
    - [3. Instance-Level Annotations](#3-instance-level-annotations)
    - [4. Captions for Images](#4-captions-for-images)
    - [5. Training, Validation, and Test Splits](#5-training-validation-and-test-splits)
    - [6. Evaluation Metrics](#6-evaluation-metrics)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
    - [Step 1: Set up the Environment](#step-1-set-up-the-environment)
    - [Step 2: Install Required Libraries](#step-2-install-required-libraries)
    - [Step 3: Download and Preprocess the COCO Dataset](#step-3-download-and-preprocess-the-coco-dataset)
    - [Step 4: Prepare the Data for Training](#step-4-prepare-the-data-for-training)
    - [Step 5: Implement the Model](#step-5-implement-the-model)
    - [Step 6: Train the Model](#step-6-train-the-model)
    - [Step 7: Perform Image Segmentation](#step-7-perform-image-segmentation)


In this tutorial, we will delve into **how to perform image segmentation using the COCO dataset** and deep learning. Image segmentation is the process of partitioning an image into multiple segments to identify objects and their boundaries. The [COCO dataset](http://cocodataset.org/#home) is a popular benchmark dataset for object detection, instance segmentation, and image captioning tasks. We will use deep learning techniques to train a model on the COCO dataset and perform image segmentation.

## COCO Dataset Overview

The COCO (Common Objects in Context) dataset is a widely used large-scale benchmark dataset for computer vision tasks, including object detection, instance segmentation, and image captioning. It provides a large-scale collection of high-quality images, along with pixel-level annotations for multiple object categories.

COCO was created to address the limitations of existing datasets, such as Pascal VOC and ImageNet, which primarily focus on object classification or bounding box annotations. COCO extends the scope by providing rich annotations for both object detection and instance segmentation.

The key features of the COCO dataset include:

### 1. Large-Scale Image Collection

The COCO dataset contains over 200,000 images, making it one of the largest publicly available datasets for computer vision tasks. The images are sourced from a wide range of contexts, including everyday scenes, street scenes, and more. The large-scale collection ensures diversity and represents real-world scenarios.

### 2. Object Categories

COCO covers a wide range of object categories, including common everyday objects, animals, vehicles, and more. It consists of 80 distinct object categories, such as person, car, dog, and chair. The variety of object categories enables comprehensive evaluation and training of computer vision models.

### 3. Instance-Level Annotations

One of the distinguishing features of the COCO dataset is its detailed instance-level annotations. Each object instance in an image is labeled with a bounding box and a pixel-level segmentation mask. This fine-grained annotation allows models to understand the boundaries and shapes of objects, making it suitable for tasks like instance segmentation.

### 4. Captions for Images

In addition to object annotations, the COCO dataset includes five English captions for each image. This aspect of the dataset makes it valuable for natural language processing tasks, such as image captioning and multimodal learning.

### 5. Training, Validation, and Test Splits

The COCO dataset is divided into three main subsets: training, validation, and test. The training set consists of a large number of images (around 118,000), which are commonly used for training deep learning models. The validation set (around 5,000 images) is used for hyperparameter tuning and model evaluation during development. The test set (around 40,000 images) is not publicly available, and its annotations are withheld for objective evaluation in benchmark challenges.

### 6. Evaluation Metrics

COCO introduces evaluation metrics tailored for different tasks. For object detection, the widely used mean Average Precision (mAP) metric is employed, which considers precision-recall curves for different object categories. For instance segmentation, the COCO dataset uses the COCO mAP metric, which considers both bounding box accuracy and segmentation quality.

Overall, the COCO dataset has become a standard benchmark for evaluating and advancing state-of-the-art computer vision models. Its large-scale image collection, detailed annotations, and diverse object categories make it a valuable resource for developing and evaluating models for various computer vision tasks.

## Prerequisites

Before getting started, make sure you have the following:

- Python 3.6 or above: Python is the programming language we'll use for the tutorial.
- TensorFlow 2.x or PyTorch: We'll use one of these deep learning frameworks for building and training the segmentation model.
- COCO dataset: Download the COCO dataset from the official website (http://cocodataset.org/#download). Choose the desired version (e.g., 2017) and download the following files:
  - Train images: train2017.zip
  - Train annotations: annotations_trainval2017.zip

After downloading, extract the contents of both ZIP files into a directory of your choice.

## Steps

### Step 1: Set up the Environment

- Create a new directory for your project and navigate to it using the terminal or command prompt.
- Set up a virtual environment (optional but recommended) to keep your project dependencies isolated.

### Step 2: Install Required Libraries

Open a terminal or command prompt and run the following command to install the necessary libraries:

```bash
pip install tensorflow opencv-python pycocotools ujason
```

We'll install TensorFlow (or PyTorch), OpenCV, and the `pycocotools` library to work with the COCO dataset.

### Step 3: Download and Preprocess the COCO Dataset

Before training a model on the COCO dataset, we need to preprocess it and prepare it for training. There are existing scripts available that automate this process. We will use the `pycocotools` library to preprocess the dataset.

Create a new Python script file (e.g., `preprocess_coco.py`) and add the following code:

```python
import os
import cv2
from pycocotools.coco import COCO
import ujson as json
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Filter the UserWarning related to low contrast images
warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

# Specify the paths to the COCO dataset files
data_dir = "./"
train_dir = os.path.join(data_dir, 'train2014')
val_dir = os.path.join(data_dir, 'val2014')
annotations_dir = os.path.join(data_dir, 'annotations')
train_annotations_file = os.path.join(annotations_dir, 'instances_train2014.json')
val_annotations_file = os.path.join(annotations_dir, 'instances_val2014.json')

# Create directories for preprocessed images and masks
preprocessed_dir = './preprocessed'
os.makedirs(os.path.join(preprocessed_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'val', 'masks'), exist_ok=True)

batch_size = 10  # Number of images to process before updating the progress bar

def preprocess_image(img_info, coco, data_dir, output_dir):
    image_path = os.path.join(data_dir, img_info['file_name'])
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
    if len(ann_ids) == 0:
        return

    mask = coco.annToMask(coco.loadAnns(ann_ids)[0])

    # Save the preprocessed image
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_dir, 'images', img_info['file_name']), image)

    # Save the corresponding mask
    cv2.imwrite(os.path.join(output_dir, 'masks', img_info['file_name'].replace('.jpg', '.png')), mask)

def preprocess_dataset(data_dir, annotations_file, output_dir):
    coco = COCO(annotations_file)
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    image_infos = coco_data['images']

    total_images = len(image_infos)
    num_batches = total_images // batch_size

    # Use tqdm to create a progress bar
    progress_bar = tqdm(total=num_batches, desc='Preprocessing', unit='batch(es)', ncols=80)

    with ThreadPoolExecutor() as executor:
        for i in range(0, total_images, batch_size):
            batch_image_infos = image_infos[i:i+batch_size]
            futures = []

            for img_info in batch_image_infos:
                future = executor.submit(preprocess_image, img_info, coco, data_dir, output_dir)
                futures.append(future)

            # Wait for the processing of all images in the batch to complete
            for future in futures:
                future.result()

            progress_bar.update(1)  # Update the progress bar for each batch

    progress_bar.close()  # Close the progress bar once finished

# Preprocess the training set
preprocess_dataset(train_dir, train_annotations_file, os.path.join(preprocessed_dir, 'train'))

# Preprocess the validation set (if required)
preprocess_dataset(val_dir, val_annotations_file, os.path.join(preprocessed_dir, 'val'))
```


Run the script to preprocess the COCO dataset:

```bash
python preprocess_coco.py
```

This script will save the preprocessed images and masks in the specified output directory.

```
loading annotations into memory...
Done (t=10.78s)
creating index...
index created!
Preprocessing: 8279batch(es) [16:15,  8.49batch(es)/s]                          
loading annotations into memory...
Done (t=8.27s)
creating index...
index created!
Preprocessing: 4051batch(es) [09:28,  7.13batch(es)/s]    
```

### Step 4: Prepare the Data for Training

Now that we have preprocessed the COCO dataset, we need to create a data pipeline to load and preprocess the data during training.

Create a new Python script file (e.g., `data_loader.py`) and add the following code:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logs
import tensorflow as tf

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    return mask

def parse_image_mask(image_path, mask_path):
    image = load_image(image_path)
    mask = load_mask(mask_path)
    return image, mask

def create_dataset(data_dir, batch_size):
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    image_paths = tf.data.Dataset.list_files(os.path.join(image_dir, '*.jpg'))
    image_paths = list(image_paths.as_numpy_iterator())  # Convert dataset to a list

    mask_paths = [os.path.join(mask_dir, os.path.basename(image_path.decode()).replace('.jpg', '.png')) for image_path in image_paths]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image_mask)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# Example usage
batch_size = 8
data_dir = './preprocessed'
train_dataset = create_dataset(os.path.join(data_dir, 'train'), batch_size)
val_dataset = create_dataset(os.path.join(data_dir, 'val'), batch_size)
```

This script provides functions to load and preprocess the images and masks, as well as create TensorFlow datasets for the training and validation sets.

### Step 5: Implement the Model

The next step is to implement the image segmentation model. There are various deep learning architectures available for image segmentation, such as U-Net, DeepLab, and Mask R-CNN. Here, we will use the U-Net architecture as an example.

Create a new Python script file (e.g., `unet_model.py`) and add the following code:

```python
import tensorflow as tf

def create_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up3 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv2), conv1], axis=-1)
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up3)
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv3)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
input_shape = (256, 256, 3)
num_classes = 2  # Background + Object
model = create_model(input_shape, num_classes)
model.summary()
```

This script defines the U-Net model architecture using TensorFlow's Keras API. Adjust the `input_shape` and `num_classes` variables according to your requirements.

### Step 6: Train the Model

Finally, we can train the image segmentation model using the preprocessed COCO dataset.

Create a new Python script file (e.g., `train.py`) and add the following code:

```python
import os
import tensorflow as tf

def train_model(train_dataset, val_dataset, model, epochs, save_dir):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, 'model_checkpoint.h5'),
        save_best_only=True,
        verbose=1
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[checkpoint_callback])

# Example usage
save_dir = '<path_to_save_directory>'
epochs = 10
model_save_dir = os.path.join(save_dir, 'model')
os.makedirs(model_save_dir, exist_ok=True)

train_model(train_dataset, val_dataset, model, epochs, model_save_dir)
```

Replace `<path_to_save_directory>` with the desired directory where you want to save the trained model.

Run the script to start the training process:

```bash
python train.py
```

The model checkpoints will be saved in the specified directory during training.

### Step 7: Perform Image Segmentation

After training the model, we can use it to perform image segmentation on new images.

Create a new Python script file (e.g., `segment_images.py`) and add the following code:

```python
import os
import tensorflow as tf

def segment_image(image_path, model, threshold=0.5):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = tf.expand_dims(image, 0)

    predictions = model.predict(image)
    mask = tf.argmax(predictions, axis=-1)
    mask = tf.squeeze(mask)

    mask = tf.where(mask >= threshold, 255, 0)
    mask = tf.cast(mask, tf.uint8)

    return mask

# Example usage
image_path = '<path_to_image>'
model_path = '<path_to_saved_model>'

model = tf.keras.models.load_model(model_path)
segmented_mask = segment_image(image_path, model)
```

Replace `<path_to_image>` with the path to the image you want to perform segmentation on, and `<path_to_saved_model>` with the path to the saved model checkpoint.

Run the script to segment the image:

```bash
python segment_images.py
```

The segmented mask will be saved as an image file.

Congratulations! You have learned how to perform image segmentation using the COCO dataset and deep learning. You can now apply these techniques to your own image segmentation projects. Feel free to experiment with different deep learning architectures, hyperparameters, and image preprocessing techniques to improve the segmentation results.