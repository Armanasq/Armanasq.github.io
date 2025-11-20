---
title: "TE-PINN: Quaternion-Based Orientation Estimation using Transformer-Enhanced Physics-Informed Neural Networks"
authors:
- admin
date: "2025-01-01T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2024-09-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "IEEE International Conference on Robotics and Automation (ICRA 2025)"
publication_short: "ICRA 2025"

abstract: This paper introduces TE-PINN, a novel transformer-enhanced physics-informed neural network for quaternion-based orientation estimation from IMU data. The proposed framework combines multi-head attention mechanisms with physics-based constraints to achieve robust attitude estimation in dynamic conditions. By embedding quaternion kinematics and rigid body dynamics directly into the loss function, TE-PINN enforces rotational dynamics consistency while leveraging the transformer's ability to capture temporal dependencies in IMU measurements. The RK4 quaternion integration with uncertainty quantification further enhances estimation reliability. Experimental results demonstrate a 36.8% reduction in attitude estimation error compared to traditional methods, with superior robustness in high-noise environments. The physics-informed approach ensures physically consistent predictions while maintaining computational efficiency suitable for real-time applications.

# Summary. An optional shortened abstract.
summary: Transformer-enhanced physics-informed neural network achieving 36.8% error reduction in quaternion-based attitude estimation from IMU data. Combines multi-head attention with physics-based constraints for robust real-time orientation estimation.

tags:
- Physics-Informed Neural Networks
- Transformer
- Attitude Estimation
- IMU
- Quaternion
- Deep Learning
- SLAM
featured: true

url_pdf: 'https://arxiv.org/pdf/2409.16214'
url_code: 'https://github.com/Armanasq/TE-PINN-Transformer-Enhanced-Physics-Informed-Neural-Network-Quaternion-Orientation-Estimation'
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'TE-PINN Architecture'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
projects: []

# Slides (optional).
slides: ""
---

## Highlights

- **36.8% Error Reduction**: Significant improvement in attitude estimation accuracy compared to traditional filtering methods
- **Transformer Architecture**: Multi-head attention mechanisms for capturing temporal dependencies in IMU data
- **Physics-Informed Learning**: Embedded quaternion kinematics and rigid body dynamics as physics-based constraints
- **Robustness**: Superior performance in high-noise and dynamic conditions
- **Real-Time Capability**: Efficient architecture suitable for real-time orientation estimation
- **Open Source**: Code and datasets publicly available for reproducibility

## Status

**Submitted to ICRA 2025** - Under Review

## Links

- [arXiv Preprint](https://arxiv.org/pdf/2409.16214)
- [GitHub Repository](https://github.com/Armanasq/TE-PINN-Transformer-Enhanced-Physics-Informed-Neural-Network-Quaternion-Orientation-Estimation)
