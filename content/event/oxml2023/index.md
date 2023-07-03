---
title: Deep Learning-based Carcinoma Classification - OxML 2023
math: true
event: Oxford Machine Learning Summer School 2023

location: Virtual
address:
  city: Oxford
  country: UK

summary:
abstract: 
# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2023-07-01T15:00:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
publishDate: '2023-07-01T15:00:00Z'

authors: []
tags: []

# Is this a featured talk? (true/false)
featured: false

image:
  caption: 
  focal_point: Right

links:
  - icon: twitter
    icon_pack: fab
    name: Follow
    url: https://twitter.com/georgecushen
url_code: 'https://www.kaggle.com/code/armanasgharpoor1993/ensemble-learning-k-fold?scriptVersionId=135323558'
url_pdf: ''
url_slides: './uploads/OXML2023.pptx'
url_video: ''

# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
#slides: oxml2023

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects:
  - oxml2023
---

# Advancing Cancer Diagnosis: Improving Classification of Histopathological Slices

As participants of the Oxford Machine Learning Summer School 2023, we are thrilled to announce our achievement in the implementation of transfer learning techniques for improving the classification of cancer cells in H&E stained histopathological slices. We are proud to have secured the 1st and 3rd positions in the Health Cases Competition.

Today, we present our project, which aimed for cancer diagnosis by identifying the presence of carcinoma cells and determining their benign or malignant nature.

Objective: Our primary goal was to develop a robust classification system capable of accurately distinguishing H&E stained histopathological slices as either containing carcinoma cells or not, while also providing insights into the nature of the carcinoma, whether it is benign or malignant.

Dataset: We utilized a dataset comprising 186 histopathological slides from breast biopsies, with 62 of them annotated with labels for training and evaluation.

Challenge: Throughout our project, we faced two main challenges. First, we had limited annotated training data, necessitating innovative approaches to ensure reliable results. Second, the dataset had an uneven distribution of classes, which required careful handling to address class imbalances.

To address these challenges, we employed transfer learning, a technique that utilizes pre-trained models on large-scale datasets to enhance performance on specific tasks. By leveraging the knowledge gained from prior tasks, we aimed to improve the accuracy of cancer cell classification in histopathological slices.

Our research involved several key stages. We meticulously preprocessed the data to optimize the input images for accurate analysis. We then selected suitable models and fine-tuned them to optimize their performance for our specific task.

Additionally, we employed techniques such as data augmentation and class balancing to mitigate the impact of limited training data and class imbalances. These approaches played a crucial role in enhancing the model's ability to generalize and make accurate predictions.

Throughout our project, we conducted thorough evaluations using established performance metrics such as accuracy, precision, recall, and F1-score. We also compared our results with state-of-the-art methods and performed extensive cross-validation to ensure the reliability and generalizability of our findings.


{{< gdocs src="https://docs.google.com/presentation/d/1bdQX9ksPfC5dzYAJoxk6FSo-sAClPRiM/edit?usp=sharing&ouid=116218384712094042101&rtpof=true&sd=true" >}}
