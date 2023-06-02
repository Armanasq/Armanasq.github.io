---
layout: archive
title: "Robotic"
permalink: /robotic/
---

<div class="section-container">
  <div class="section">
    <a href="/ros/">
      <div class="section-content">
        <img src="/ros.png" alt="ROS">
        <h2>Robotic Operating System <br> (ROS)</h2>
      </div>
    </a>
  </div>
  
  <div class="section">
    <a href="/deep-learning/">
      <div class="section-content">
        <img src="/deep-learning-image.png" alt="Deep Learning">
        <h2>Deep Learning</h2>
      </div>
    </a>
  </div>

  <div class="section">
    <a href="/orbital-mechanics/">
      <div class="section-content">
        <img src="/orbital-mechanics.png" style="width:120px; margin:0 Auto;" alt="Orbital Mechanics">
        <h2>Orbital Mechanics</h2>
      </div>
    </a>
  </div>

  <div class="section">
    <a href="/attitude-estimation/">
      <div class="section-content">
        <img src="/attitude.png" alt="Attitude Estimation">
        <h2>Attitude Estimation</h2>
      </div>
    </a>
  </div>
  
  <div class="section">
    <a href="/odometry/">
      <div class="section-content">
        <img src="/odometry.png" alt="Odometry">
        <h2>Odometry</h2>
      </div>
    </a>
  </div>
  
  <div class="section">
    <a href="/datasets/">
      <div class="section-content">
        <img src="/datasets.png" alt="Datasets">
        <h2>Datasets</h2>
      </div>
    </a>
  </div>

  <div class="section">
    <a href="/compute-vision/">
      <div class="section-content">
        <img src="/computer-vision.png" style="width:120px; margin:0 Auto;" alt="Computer Vision">
        <h2>Computer Vision</h2>
      </div>
    </a>
  </div>

  <div class="section">
    <a href="/kaggle/">
      <div class="section-content">
        <img src="/Kaggle_logo.png" style="width:125px; margin:25px Auto; padding:10px;" alt="Kaggle">
        <h2 style="margin">Kaggle</h2>
      </div>
    </a>
  </div>
</div>

<style>
  .section-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    grid-gap: 20px;
    margin-top: 50px;
  }
  
  .section {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 250px;
    background-color: #f5f5f5;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
  }
  
  .section:hover {
    box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2);
    background-color: transparent;
  }
  
  .section-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #333;
    transition: all 0.3s ease;
    padding: 20px;
    border-radius: 10px;
    width: 100%;
  }
  
  .section:hover .section-content {
    background-color: transparent;
  }
  
  .section img {
    width: 150px;
    height: auto;
    margin-bottom: 20px;
    transition: all 0.3s ease;
  }
  

  
  .section h2 {
    font-size: 20px;
    font-weight: 500;
    margin: 0;
    transition: all 0.3s
  }
    .section:hover h2 {
    transform: scale(1.2);
    color: #fff;
    background-color: #048aff;
    margin: 10px;
    padding: 8px;
    border-radius: 5px;
  }
  .section:hover img {
    box-shadow: 0px 5px 10px rgba(0, 0, 0, 0);
    background-color: transparent;
  }
</style>
