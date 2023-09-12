---
title: "ROS Tutorial 1: Basic Concepts "
date: 2023-01-02
description: "Introduction"
url: "/ros/tutorial-01/"
showToc: true
math: true
disableAnchoredHeadings: False
commentable: true
tags:
  - ROS
  - Tutorial
---
[&lArr; ROS](/ros/)
<img src="/ros.png" alt="ROS" style="width:350px;display: block;
  margin-left: auto;
  margin-right: auto; margin-top:0px auto" >
</div>

# ROS Tutorial 1: Basic Concepts 

In this tutorial, we'll introduce you to the fundamental concepts of the Robot Operating System (ROS) without diving into complex jargon. We'll start from scratch and gradually build our understanding of ROS.


## What is ROS?

ROS, the Robot Operating System, is a powerful framework for building robot software. It simplifies the development of robotic applications by providing a structured way to create, manage, and connect software components. ROS is widely adopted in the robotics community due to its open-source nature and rich ecosystem.

## Key ROS Concepts

### Nodes

ROS is designed as a distributed system of nodes. Think of nodes as small software modules that perform specific tasks, such as reading sensor data, processing information, or controlling motors. Each node communicates with others by sending and receiving messages through topics.

### Topics

Topics are named channels through which nodes exchange data. A node can publish data to a topic, and other nodes can subscribe to that topic to receive the data. Topics enable modular and decoupled communication between nodes.

### Publishers and Subscribers

- **Publisher**: A node that sends data (messages) on a topic is a publisher. Publishers broadcast information to anyone interested in that topic.

- **Subscriber**: A node that receives data (messages) from a topic is a subscriber. Subscribers listen to topics and react to the data they receive.

## Setting up the ROS Environment

Let's get started with ROS by setting up a basic development environment.

1. **Installation**: If you haven't already installed ROS, follow the installation instructions for your specific platform on the official [ROS website](http://www.ros.org/install/).

2. **Initialize ROS**: After installation, initialize ROS in your current terminal session:

   ```bash
   $ source /opt/ros/your-ros-version/setup.bash
   ```

   Replace `your-ros-version` with the ROS version you've installed.

3. **Create a Workspace**: Create a workspace to organize your ROS projects:

   ```bash
   $ mkdir -p ~/ros_workspace/src
   $ cd ~/ros_workspace/src
   $ catkin_init_workspace
   $ cd ~/ros_workspace
   $ catkin_make
   ```

## Creating Your First ROS Package

ROS organizes code into packages. Let's create a simple package named `my_first_package`:

```bash
$ cd ~/ros_workspace/src
$ catkin_create_pkg my_first_package rospy
```

Here, we're creating a Python-based ROS package (`rospy`) called `my_first_package`. This package will use Python for programming.

## Writing a Simple ROS Node

Now, let's create a basic ROS node that publishes a message to a topic. Create a Python script, e.g., `simple_publisher.py`, inside the `my_first_package` folder:

```python
#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def main():
    rospy.init_node('simple_publisher')
    pub = rospy.Publisher('my_topic', String, queue_size=10)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        message = "Hello, ROS!"
        pub.publish(message)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
```

In this script, we:

- Import necessary libraries.
- Initialize the ROS node.
- Create a publisher on the topic `my_topic`.
- Continuously publish the message "Hello, ROS!" at a rate of 1 Hz.

## Running the ROS Node

To run the ROS node, open a terminal and navigate to your workspace:

```bash
$ cd ~/ros_workspace
```

Build your workspace:

```bash
$ catkin_make
```

Now, you can run the ROS node:

```bash
$ rosrun my_first_package simple_publisher.py
```

## Checking Published Data

To verify that your node is publishing messages correctly, open another terminal and use the `rostopic echo` command:

```bash
$ rostopic echo my_topic
```

You should see the "Hello, ROS!" message being displayed.

## Conclusion

Congratulations! You've created your first ROS package and node while gaining a better understanding of ROS basics. In the next tutorial, we'll explore more advanced topics like creating custom messages and building more complex robot behaviors. Stay tuned for more ROS adventures!