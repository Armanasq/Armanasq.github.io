---
title: "Getting Started with ROS (Robot Operating System)"
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

- [Tutorial: Getting Started with ROS (Robot Operating System)](#tutorial-getting-started-with-ros-robot-operating-system)
  - [Introduction](#introduction)
  - [What is ROS?](#what-is-ros)
  - [Why ROS?](#why-ros)
  - [Step 1: Installing ROS](#step-1-installing-ros)
  - [Step 2: Creating a ROS Workspace](#step-2-creating-a-ros-workspace)
  - [Step 3: Creating a ROS Package](#step-3-creating-a-ros-package)
  - [Step 4: Writing a ROS Program](#step-4-writing-a-ros-program)
  - [Conclusion](#conclusion)



# Tutorial: Getting Started with ROS (Robot Operating System)

## Introduction
Welcome to our step-by-step tutorial on getting started with ROS (Robot Operating System). ROS is an open-source framework for building robotic systems. It provides a collection of libraries, tools, and conventions to help developers create robust and modular robot applications. In this tutorial, we will guide you through the process of setting up ROS, creating a ROS workspace, and running your first ROS program. By the end of this tutorial, you will have a solid foundation for working with ROS and developing your own robotic applications.

## What is ROS?

ROS, short for Robot Operating System, is an open-source framework designed for building robotic systems. It provides a flexible and modular architecture that enables developers to create complex robot applications by leveraging a wide range of libraries, tools, and community-contributed packages. 

ROS was initially developed at Stanford University in 2007 and has since gained significant popularity in the robotics community. It has a large and active user base, which has contributed to the extensive development and refinement of its features and capabilities.

Despite its name, ROS is not an operating system in the traditional sense. Rather, it serves as a middleware layer that runs on top of a conventional operating system (such as Linux) and provides a set of abstractions and functionalities specifically tailored for robotics.

One of the key strengths of ROS is its focus on collaboration and reusability. It encourages the development and sharing of reusable software components called "packages." These packages encapsulate specific functionalities or algorithms, making it easier for developers to build upon existing work and leverage the collective knowledge and expertise of the ROS community.

ROS follows a distributed architecture, where different processes, called "nodes," communicate with each other by passing messages. This messaging system allows nodes to exchange data, commands, and sensor information in a standardized and interoperable manner. It promotes modularity and scalability, enabling developers to break down complex systems into smaller, manageable components that can be developed and tested independently.

Furthermore, ROS provides a wide range of tools for visualization, simulation, debugging, and analysis, which greatly simplify the development and debugging process. These tools include visualization tools like RViz for visualizing robot models and sensor data, simulation environments like Gazebo for testing and evaluating robot behavior, and debugging tools like rqt_console for monitoring and analyzing the system's log messages.

Overall, ROS has become the de facto standard in the field of robotics due to its versatility, modularity, and active community. It has been widely adopted in both academic and industrial settings for a variety of applications, ranging from autonomous vehicles and industrial robots to medical robotics and research platforms.

In the next sections of this tutorial, we will guide you through the process of setting up ROS, creating a workspace, and running your first ROS program. This will provide you with a solid foundation to start developing your own robotic applications using the powerful capabilities of ROS.

## Why ROS?

ROS, or Robot Operating System, has gained significant popularity in the robotics community for several compelling reasons. Let's explore why ROS has become the framework of choice for many roboticists and researchers:

1. **Modularity and Reusability:** ROS encourages a modular approach to software development, where functionalities are encapsulated into reusable components called packages. This modularity makes it easier to develop, test, and maintain individual components, and enables seamless integration of different software modules into a larger robotic system. The extensive package ecosystem of ROS allows developers to leverage existing solutions and build upon the work of others, saving time and effort.

2. **Interoperability:** ROS promotes interoperability by providing a standardized messaging system for communication between different nodes. Nodes can exchange data, commands, and sensor information using ROS messages, services, and topics. This standardized communication protocol allows for the seamless integration of various hardware and software components, making it easier to build complex robotic systems with heterogeneous components.

3. **Community and Collaboration:** ROS has a large and active community of developers and researchers. This vibrant community contributes to the continuous improvement and evolution of ROS through the development of new packages, bug fixes, and documentation. The ROS community also provides support through forums, mailing lists, and user groups, making it easier for newcomers to get started and seek help when needed.

4. **Visualization and Simulation Tools:** ROS provides a rich set of tools for visualization, simulation, and analysis. Tools like RViz allow developers to visualize robot models, sensor data, and planning algorithms in a 3D environment. Simulation environments like Gazebo enable the realistic simulation of robots and their interactions with the environment, facilitating the testing and evaluation of robot behavior before deploying on physical hardware. These tools aid in debugging, performance analysis, and visualization of complex robotic systems.

5. **Robustness and Scalability:** ROS is designed to be robust and scalable, allowing for the development of complex robotic systems. With its distributed architecture, ROS supports the deployment of multiple nodes across different machines, enabling parallel processing and distributed computing. This scalability is crucial for applications that require real-time processing or involve large-scale robot networks.

6. **Open-Source and Cross-Platform:** ROS is an open-source framework, which means it is freely available and can be modified and redistributed. This open nature has fostered a collaborative environment where researchers and developers can share their work, contribute improvements, and build upon existing projects. ROS also supports cross-platform development, allowing developers to work with different operating systems such as Linux, macOS, and Windows.

7. **Education and Learning:** ROS has become a popular choice for educational institutions and learning resources. Many universities and research institutions incorporate ROS into their robotics curricula, providing students with hands-on experience in developing robotic systems. The availability of tutorials, documentation, and online courses makes it easier for beginners to learn ROS and apply it to their own projects.

These factors, among others, have contributed to the widespread adoption of ROS in the robotics community. Whether you are a researcher, a student, or a hobbyist, ROS offers a powerful and flexible framework for developing innovative and complex robotic systems. By leveraging the benefits of ROS, you can focus on the high-level logic and algorithms of your robot, while benefiting from the extensive resources and community support that ROS provides.

## Step 1: Installing ROS
The first step is to install ROS on your system. ROS supports various Linux distributions, with Ubuntu being the most commonly used. To install ROS, you need to follow these steps:

1. **Choose ROS Distribution:** Determine which ROS distribution is compatible with your operating system. The two most recent distributions are ROS Noetic for Ubuntu 20.04 and ROS Melodic for Ubuntu 18.04. Choose the appropriate distribution for your system based on compatibility and community support.

2. **Installation:** Visit the official ROS website at http://www.ros.org and follow the installation instructions provided for your chosen distribution. The installation process typically involves running a set of commands in the terminal to set up the ROS repositories and install the necessary packages. It is recommended to install the full desktop version, which includes the core ROS packages as well as commonly used tools and libraries.

3. **Environment Setup:** After installation, set up the ROS environment by sourcing the appropriate setup file. This step ensures that ROS commands and tools are available in your terminal. Open a terminal and run the following command:

   ```bash
   source /opt/ros/<ros-distro>/setup.bash
   ```

   Replace `<ros-distro>` with the name of your ROS distribution (e.g., noetic or melodic). You can add this command to your shell's initialization file (e.g., `.bashrc`) to automatically set up the environment each time you open a new terminal.

## Step 2: Creating a ROS Workspace
Once ROS is installed, the next step is to create a ROS workspace. A workspace is a directory that organizes your ROS packages and provides a build system for compiling and managing them. To create a ROS workspace, follow these steps:

1. **Create Workspace Directory:** Decide on a location for your ROS workspace directory. For example, you can create a directory named `catkin_ws` in your home directory.

2. **Initialize Workspace:** Open a terminal and navigate to the directory you created. Run the following command to initialize the workspace:

   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/
   catkin_make
   ```

   This command creates a `src` directory inside the workspace, which is where you will place your ROS packages. It also generates the necessary build and configuration files for the workspace.

## Step 3: Creating a ROS Package
With the workspace set up, you can now create a ROS package to start developing your robot applications. A package is a directory that contains ROS nodes, libraries, configuration files, and other resources. To create a ROS package, follow these steps:

1. **Navigate to Source Directory:** Open a terminal and navigate to the source directory of your ROS workspace:

   ```bash
   cd ~/catkin_ws/src
   ```

2. **Create Package:** Run the following command to create a ROS package named `my_package`:

   ```bash
   catkin_create_pkg my_package
   ```

   This command generates a package directory with the specified name and sets up the necessary package configuration files. You can customize the package by adding dependencies and other metadata in the

 generated `package.xml` file.

3. **Build Workspace:** After creating the package, navigate back to the root of the workspace (`~/catkin_ws/`) and run `catkin_make` again to build the workspace with the new package included:

   ```bash
   cd ~/catkin_ws/
   catkin_make
   ```

   This command compiles the packages in the workspace and generates the necessary executables and libraries.

## Step 4: Writing a ROS Program
Now that you have a ROS package, you can start writing your first ROS program. In ROS, programs are organized as nodes, which are independent processes that communicate with each other through messages, services, and other communication mechanisms. To create and run a ROS program, follow these steps:

1. **Navigate to Package Directory:** Open a terminal and navigate to the directory of your ROS package:

   ```bash
   cd ~/catkin_ws/src/my_package
   ```

2. **Add Source Files:** Inside the package directory, add your source files. For example, you can create a simple Python script named `my_node.py` that publishes a ROS message. In this script, you can import the necessary ROS libraries, define a publisher, and publish messages on a specific topic.

3. **Build Workspace:** After adding the source files, navigate back to the root of the workspace (`~/catkin_ws/`) and run `catkin_make` again to build the workspace with the updated package:

   ```bash
   cd ~/catkin_ws/
   catkin_make
   ```

4. **Run ROS Master:** To start the ROS communication infrastructure, open a terminal and run the following command to launch the ROS Master:

   ```bash
   roscore
   ```

   The ROS Master is responsible for managing the communication between different ROS nodes.

5. **Run ROS Node:** Open another terminal, navigate to the package directory (`~/catkin_ws/src/my_package`), and run the ROS node using the following command:

   ```bash
   rosrun my_package my_node.py
   ```

   This command executes the ROS node and starts publishing messages according to the logic defined in your script.

6. **Verify Output:** To verify that the ROS node is running and publishing messages, open a new terminal and run the following command:

   ```bash
   rostopic echo /my_topic
   ```

   Replace `/my_topic` with the actual topic name used in your script. This command subscribes to the specified topic and displays the published messages in the terminal.

## Conclusion
Congratulations on completing the tutorial on getting started with ROS! You have learned how to install ROS, create a ROS workspace, create a ROS package, and run your first ROS program. This tutorial provides a solid foundation for working with ROS and developing your own robotic applications. With ROS, you have access to a wide range of tools and libraries that simplify the development process and enable you to build complex robotic systems. Make sure to explore the official ROS documentation and community resources to expand your knowledge and dive deeper into the world of robotics with ROS. Happy coding and robot building!