---
title: "Generalizable end-to-end deep learning frameworks for real-time attitude estimation using 6DoF inertial measurement units"
authors:
- admin
- Mohammad H. Sabour
author_notes:
- "Equal contribution"
- "Equal contribution"
date: "2023-04-01T00:00:00Z"
doi: "10.1016/j.measurement.2023.113105"

# Schedule page publish date (NOT publication's date).
publishDate: "2023-04-31T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "*Journal of the International Measurement Confederation"
publication_short: "Measurement"

abstract: This paper presents a novel end-to-end deep learning framework for real-time inertial attitude estimation using 6DoF IMU measurements. Inertial Measurement Units are widely used in various applications, including engineering and medical sciences. However, traditional filters used for attitude estimation suffer from poor generalization over different motion patterns and environmental disturbances. To address this problem, we propose two deep learning models that incorporate accelerometer and gyroscope readings as inputs. These models are designed to be generalized to different motion patterns, sampling rates, and environmental disturbances. Our models consist of convolutional neural network layers combined with Bi-Directional Long–Short Term Memory followed by a Fully Forward Neural Network to estimate the quaternion. We evaluate the proposed method on seven publicly available datasets, totaling more than 120 h and 200 kilometers of IMU measurements. Our results show that the proposed method outperforms state-of-the-art methods in terms of accuracy and robustness. Additionally, our framework demonstrates superior generalization over various motion characteristics and sensor sampling rates. Overall, this paper provides a comprehensive and reliable solution for real-time inertial attitude estimation using 6DoF IMUs, which has significant implications for a wide range of applications.

# Summary. An optional shortened abstract.
summary: • End-to-end learning framework for real-time inertial attitude estimation.
 Generalized across various sampling rates.
 RNN-CNN networks employed to learn motion characteristics, noise, and bias.
 Proposed approach outperforms traditional algorithms and other deepOutperforms traditional algorithms in terms of accuracy up to 40
 Evaluated using seven datasets, totaling 120 h and 200 kilometers of IMU measurements.

tags:
- Deep Learning
- Attitude Estimation
- IMU
- Navigation
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: https://authors.elsevier.com/a/1hCidxsQaMHxb
url_code: 'https://github.com/Armanasq/End-to-End-Deep-Learning-Framework-for-Real-Time-Inertial-Attitude-Estimation-using-6DoF-IMU'
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ''
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
---

{{% callout note %}}
Click the *Cite* button above to import publication metadata.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the *Slides* button to check out the example.
{{% /callout %}}

[Model A](Model_A.png)
Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/).

<img class="myImg" src="Model_A.png" alt="Model A">

<img class="myImg" src="Model_B.png" alt="Model B">

<img class="myImg" src="Model_C.png" alt="Model C">

