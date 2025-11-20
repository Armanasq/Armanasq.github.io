---
layout: archive
title: "Experiences"
url: /experiences/
---

# Professional Experience

## AI-Ark
*Dec 2024 to Oct 2025*

*Role: Lead Machine Learning Engineer*

Leading enterprise AI solutions development at [app.ai-ark.com](https://app.ai-ark.com/)

**Core Infrastructure:**
- Architected hybrid vector search system with Nomic-Embed-1.5B and Vespa/Elasticsearch
- Optimized ANN/KNN ranking profiles improving F1-score from 0.76 to 0.92 on benchmarks
- Designed Oracle DB integration pipeline processing 5M+ documents daily with BGE-m3

**Model Development:**
- Fine-tuned T5, BART, and Qwen models on information extraction tasks
- Implemented multi-label classifications with BERT, reaching 92% and 86% F1-score across 15 and 9 categories
- Implemented 4-bit and 8-bit quantization using GGUF and AWQ techniques, reducing model size by 75% while maintaining +90% of performance

**Data Engineering:**
- Automated domain-specific data collection pipeline using ZenRows and AWS Bedrock
- Reduced data preparation time by 65%

### Multi-Turn Multi-User RAG Chatbot with Persistent Memory for Customer Support
- Architected conversational memory system using Qdrant vector database with thread-scoped session management and buffer memory for context-aware multi-turn interactions
- Implemented session persistence layer with Redis for storing conversation history, user interaction metadata, and maintaining state across distributed instances
- Deployed hybrid search infrastructure in Qdrant combining dense vector embeddings, sparse vectors (BM25), and TF-IDF indexing for knowledge base and FAQ retrieval
- Integrated HuggingFace Text Embeddings Inference (TEI) for production-grade embedding model deployment with optimized inference and dynamic batching

### Financial Behavioral Analysis Systems | [GitHub](https://github.com/Armanasq/Financial-Behavioral-Classification-System)
- Led team of 6 engineers in developing classification systems for financial behavior analysis
- Designed classifier architecture achieving 95.2% pattern detection accuracy
- Built dual-classifier system (LLM & ensembles), achieving 95.2% pattern detection accuracy
- Developed nearest-neighbor engine using BAAI/bge-m3 embeddings, attaining 100% precision in financial classification
- Optimized voting classifier integrating XGBoost, RandomForest, and SVM with enhanced hyperparameter tuning
- Created production-grade feature pipeline handling complex financial ratios and missing data

## Fanaavaran Farayand Farda
*Jun 2024 to Dec 2024*

*Role: Lead AI Engineer* | [https://fffarda.ir](https://fffarda.ir)

### Retail Analytics - Ofogh360 | [Website](https://fffarda.ir/ofogh360)
- Engineered custom dataset generation pipeline with selection criteria, guided by distribution-aware EDA and decision boundary analysis
- Implemented transfer learning from YOLO-SKU110K for pseudo-labeling
- Developed data synthesis algorithm resolving class imbalance using generative augmentation and minority class oversampling techniques
- Applied few-shot learning methodology to train detection/segmentation model on 53 classes using only 204 annotated images, achieving 81% mAP
- Created custom data augmentation pipeline, reducing labeled data requirements by 40%
- Integrated transfer learning, few-shot techniques, and synthesized data into end-to-end training workflow, improving product recognition accuracy by 32% across 1,000+ SKUs
- Reduced model latency by 70% through INT8 quantization and TensorRT conversion, enabling real-time inference (50ms)
- Deployed models via TorchServe with custom handlers, supporting 200+ concurrent requests

### Page Streaming Segmentation & Document Intelligence Platform
- Fine-tuned multi-lingual vision-language models for invoices OCR
- Built training pipeline with 5K+ annotated documents using Roboflow and Hugging Face datasets
- Developed Streamlit UI for real-time parsing, reduced processing time from 2 mins to 30 secs

## Fasta Robotics
*Nov 2023 to Oct 2025*

*Role: AI Researcher (Part-time)*

### Autonomous Warehouse Robotic Systems
- Implemented multi-robot coordination framework using ORB-SLAM3 visual-inertial mapping system on ROS2 using Intel RealSense D435i and ZED 2i cameras
- Developed VIO-LiDAR sensor fusion pipeline achieving 15cm localization accuracy in dynamic warehouse environments using Jetson Orin NX compute platform
- Deployed YOLOv12 object detection achieving 95% pedestrian detection accuracy for real-time collision avoidance
- Designed end-to-end learning framework for inertial odometry estimation, improving robustness against sensor degradation by 30% through adaptive IMU sampling rate optimization
- Integrated heterogeneous sensor streams (visual, inertial, LiDAR) for collaborative mapping and navigation decision-making across robot teams
- Implemented sensor abstraction layer for Visual-Inertial and LiDAR streams (KITTI format)
- Engineered modular VIO-LiDAR fusion framework with real-time point cloud registration
- Developed extensible SLAM backend with automated performance benchmarking

## Rajaei Cardiovascular Center
*May 2022 to Jun 2024*

*Role: Machine Learning Engineer*

### ResearchPulse: Literature Intelligence
- Built search architecture achieving 92% precision across PubMed/Scopus databases
- Created visualization tools for trends analysis
- Deployed a contextual QA, evidence-based chatbot reducing literature review time
- Developed a Scopus/PubMed search system with trend visualization and QA chatbot
- Managed team of 5 ML engineers in developing automated classification pipelines

### Clinical Decision Framework
- Built RAG pipeline processing 200+ cases with 87% diagnostic accuracy
- Developed NLP components standardizing clinical narratives into outputs
- Integrated clinical guidelines into inference with traceable reasoning

### Medical Coding Platform
- Designed validation system reducing ICD-10 coding efforts by 75% at 94% accuracy
- Built validation against guidelines, eliminating critical misclassifications

### LLM Research
- Implemented hybrid retrieval system combining BM25, FAISS, and cross-encoder reranking
- Implemented semantic chunking with sliding window approach, enhancing contextual preservation for long-document understanding
- Tackled hallucinations issue with multi-routing and prompt engineering
- Fine-tuned Nemotron-3 and AYA-101 models on Casual Language Modeling and Seq-2-Seq
- Utilized instruction fine-tuning on LLAMA 3 to improve instruction following

### Conversational RAG Chatbot
- Conducted feasibility study evaluating 15 LLMs (7B-70B parameters) on local deployment using Ollama, benchmarking response quality and latency
- Implemented hierarchical document chunking strategy for 150K+ pages across 11K topics, optimizing for context preservation
- Created hybrid vector store using FAISS (HNSW index) with BGE-M3 embeddings and BM25 sparse vectors
- Developed multi-route orchestration system with dynamic LLM selection based on query complexity, token budget, and specialized knowledge domains
- Engineered custom prompt templates with few-shot examples and structured output formatting for consistent response generation
- Implemented query decomposition and Fusion RAG for handling vague and edge cases to improve retrieval accuracy
- Integrated cross-encoder reranking to improve relevance scoring and improve final generation accuracy
- Built API gateway interfacing with OpenAI, Anthropic, Google (Gemini), and local Ollama endpoints with automatic failover mechanisms
- Implemented Flask-based backend with streaming responses and rate limiting
- Added user feedback collection mechanism to continuously fine-tune retrieval parameters and improve answer quality through supervised learning

### BiodataBank Clinical Platform - [NCIBB.ir](https://develop.ncibb.ir/)
- Led cross-functional team of medical professionals and engineers, launching platform 2 weeks ahead of schedule and 15% under budget
- Architected RAG system for 100K+ clinical documents information retrieval system
- Optimized vector search latency by 70%, reducing average query time from 2.1s to 0.6s
- Developed differential diagnosis engine on validated medical case studies across 20+ specialties
- Created real-time research visualization system analyzing PubMed papers with QA capabilities
- Built multi-route LLM-based automated ICD-10 coding system with 91% accuracy, reducing manual coding time by 75%
- Created LLM-based explainable AI module highlighting evidence supporting diagnostic suggestions

### Medical Case Generator
- Created generative AI system for clinical scenario simulation with parameter controls
- Integrated feedback mechanisms for medical education research

### Automated ICD-10 Labeler
- Developed hybrid retrieval system combining sparse (BM25) and dense (FAISS) embeddings
- Implemented NLP pipeline for medical entity extraction and ontology mapping
- Engineered sliding window chunking algorithm with cross-encoder reranking
- Created validation framework interfacing with ICD-10 taxonomies

## Farzan Research Institute
*Oct 2021 to Jun 2024*

*Role: AI Research Scientist*

### Evidence-based RAG Pipeline for Scientific AI-Assistant
- Fine-tuned Nemotron, AYA-101 and BERT models using LoRA for mental health analysis
- Created comprehensive LLM evaluation framework with domain-specific metrics
- Optimized training on A100/H100 GPUs using DeepSpeed ZeRO-3
- Engineered hybrid retrieval (BM25, FAISS, cross-encoders) and semantic chunking
- Devised Chain-of-Thought, ReAct reasoning prompt templates
- Fine-tuned Nemotron-3 8B & AYA-101 (QLoRA, 4-bit quantization) on A100 clusters
- Implemented gradient accumulation and mixed precision training on distributed A100 clusters

### AI Engineer Projects
- Built conversational QA system using RAG for CRM, reducing response time 45%
- Fine-tuned BERT/LLAMA using QLoRA improving accuracy 30%
- Built RAG-based QA chatbot using OpenAI API
- Automated FAQs and integrated shopping cart tracking for enhanced user experience

## Computer Vision Engineer & Freelancer
*Apr 2020 to Oct 2021*

*Role: Independent Contractor*

- Implemented LSTM-based image captioning achieving BLEU score of 0.32
- Built CNN-LSTM handwriting recognition with 5.2% CER and 8.1% WER on IAM database
- Developed classification models using EfficientNet with 94% accuracy on Breast Cancer Wisconsin dataset
- Created Mask R-CNN instance segmentation achieving 43.5 mAP on COCO dataset | [Kaggle Tutorial](https://www.kaggle.com/code/armanasgharpoor1993/coco-dataset-tutorial-image-segmentation)
- Developed PyTorch training pipeline with distributed computing and custom medical image augmentation techniques
- Built real-time ROS perception system for autonomous navigation with OpenCV | [GitHub](https://github.com/Armanasq/ros_robot_opencv)
- Published computer vision tutorials and documentation | [Example](https://armanasq.github.io/computer-vision/image-segementation/)
- Developed PyTorch training pipelines with distributed computing, medical image augmentation, TensorRT optimization | [PyTorch Tutorial](https://github.com/Armanasq/Deep-Learning-Tutorial/blob/main/PyTorch/Deep_Neural_Network_Implementation_Using_PyTorch.ipynb)
- Built ROS2 perception systems (depth estimation, object tracking) with KITTI dataset | [KITTI Tutorial](https://github.com/Armanasq/kitti-dataset-tutorial) | [ROS & OpenCV](https://github.com/Armanasq/ros_robot_opencv)
- Created advanced Python/CV tutorials | [Python Tutorial](https://github.com/Armanasq/Python-Programming-Starter) | [Segmentation Example 1](https://armanasq.github.io/computer-vision/image-segementation/) | [Segmentation Example 2](https://armanasq.github.io/computer-vision/image-segementation-coco/)

---

# Research Experience

## Students' Scientific Research Center
*Jul 2024 to Sep 2024*

*Role: Physics-Informed AI Researcher*

### Transformer-Enhanced PINN for Attitude Estimation
- Built TE-PINN: transformer-physics network for IMU orientation estimation
- Designed multi-head attention for IMU data, reduced error 36.8%
- Implemented RK4 quaternion integration with uncertainty quantification
- Developed PINNs with multi-head attention, reducing attitude estimation error by 36.8%
- Embedded quaternion kinematics, rigid body dynamics, and multi-head attention
- Designed a physics-based loss enforcing rotational dynamics with angular velocity and forces
- Achieved 36.8% error reduction and robustness in high-noise, dynamic conditions
- Research selected for ICRA 2025; code and datasets publicly available | [GitHub](https://github.com/Armanasq/TE-PINN-Transformer-Enhanced-Physics-Informed-Neural-Network-Quaternion-Orientation-Estimation.git) | [arXiv](https://arxiv.org/pdf/2409.16214)

## MIT IAIFI AI+Physics Program
*May 2024 to Jun 2024*

*Role: Research Participant*

### Simulation-Based Inference for Medical Analysis
- Developed Vision Transformer-based tumor severity simulator for histopathological images
- Applied Sequential Neural Posterior Estimation (SNPE) for posterior distribution analysis
- Created noise-injection model for tumor severity levels with likelihood-free inference methods
- Developed Vision Transformer-based analysis system achieving 87% diagnostic accuracy on medical imaging datasets
- Implemented Sequential Neural Posterior Estimation reducing uncertainty bounds
- Created robust uncertainty quantification framework improving confidence calibration
- Presented research findings at cross-institutional workshop with 75+ participants from MIT, Harvard, and other institutions

## Oxford Machine Learning Summer School
*May 2023 to Aug 2023*

*Role: Competition Winner*

### Vision-based Cancer Detection
- Implemented transfer learning pipeline achieving 82% accuracy, ranking 1st among 120+ participants in international competition
- Developed ensemble approach combining EfficientNet and Vision Transformer models
- Published methodology and results on GitHub and Kaggle | [GitHub](https://github.com/Armanasq/The-Health-and-Medicine-OxML-competition-track)
- Applied transfer learning to medical imaging, achieving 82% accuracy
- Ranked 1st in The Health and Medicine OxML competition track | [Leaderboard](https://www.kaggle.com/competitions/oxml-carinoma-classification/leaderboard?tab=public) | [Slides](https://armanasq.github.io/talk/deep-learning-based-carcinoma-classification-oxml-2023/)

## University of Tehran
*Sep 2019 to Sep 2022*

*Role: Graduate Researcher*

### Deep Learning for Inertial Navigation Systems
- Developed learning-based models for real-time inertial attitude estimation | [GitHub](https://github.com/Armanasq/End-to-End-Deep-Learning-Framework-for-Real-Time-Inertial-Attitude-Estimation-using-6DoF-IMU) | [Publication](https://www.sciencedirect.com/science/article/pii/S0263224123006693)
- Developed end-to-end learning models reducing attitude estimation error by 40% across 7 publicly available datasets
- 40% improvement over traditional methods on 7 datasets (100+ km IMU measurements)
- Published results in Measurements journal (Elsevier) | [IF 5.2](https://www.sciencedirect.com/journal/measurement)
- Open-sourced implementation with documentation and examples [+33 Stars] | [GitHub](https://github.com/Armanasq/End-to-End-Deep-Learning-Framework-for-Real-Time-Inertial-Attitude-Estimation-using-6DoF-IMU)

---

# Teaching & Academic Service

## Tehran University of Medical Sciences
*Sep 2022 to Present*

*Role: Co-Instructor*

**Course:** Application of Technology in Research
- Designing and taught graduate-level courses in advanced search techniques, providing rigorous assessment and guidance to cultivate deep subject matter expertise
- Conducting office hours, fostering academic excellence and professional growth

## Students' Scientific Research Center
*Sep 2022 to Present*

*Role: Instructor*

**Graduate and Undergraduate (B.Sc., M.D., M.Sc. and Ph.D.): +100 Students**
- Developing and delivered courses emphasizing practical applications of AI and Programming
- Offering personalized guidance to students, cultivating the next generation of independent researchers

## Students' Scientific Research Center
*May 2023 to Present*

*Role: Supervisor*

- Guiding 10+ students in creating AI medical imaging tools for early and accurate disease detection, enhancing patient outcomes, and cutting healthcare costs
- Leading team in six systematic reviews on AI-powered Medical Imaging Analysis, uncovering critical insights for cancer detection and time-series forecasting advancements, fostering life-saving interventions

## Students' Scientific Research Center
*Apr 2019 to Present*

*Role: Referee of Research Council*

- Shaped impactful medical science and healthcare solutions through analyzing and assessing research proposals

## University of Tehran
*Sep 2021 to Sep 2022*

*Role: Teaching Assistant*

**Course:** Fuzzy Logic (Graduate Level)
**Instructor:** Dr. M.H. Sabour
- Designed and supervised projects, enhancing programming skills for 15+ graduate students
- Enhanced academic and professional growth through focused office hours, optimizing graduate students' development with strategic support

## Aviation Industry Training Center
*Sep 2019 to Sep 2021*

*Role: Thesis Supervisor*

**Supervised 5 Undergraduate Theses:**
- Design and Implementation of a 3 Axis CNC Machine (Spring 2021 - Fall 2021)
- Design and Implementation of Pulse Circuits Training Board (Fall 2020 - Fall 2021)
- Design, Simulation, and Building of an Aircraft Fire Extinguishing System (Spring 2020 - Fall 2020)
- Design and Implementation of Retractable Landing Gear (Fall 2019 - Spring 2020)
- Design and Implementation of a CNC Hot Wire (Fall 2019 - Spring 2020)

## Aviation Industry Training Center
*Sep 2018 to Sep 2021*

*Role: Instructor*

- Instructed 11 courses on electronics, navigation, and aviation for 150+ undergraduate students
- Delivering high-caliber education, nurturing a pipeline of highly skilled aerospace professionals

---

# Review Activity

*Role: Journal & Conference Reviewer (100+ Papers Reviewed)*

Full list available at [ORCID](https://orcid.org/0000-0001-6271-4533) and [Web of Science](https://www.webofscience.com/wos/author/record/IAN-3152-2023)

## Outstanding Recognition
- **Outstanding Reviewer**, IEEE Transactions on Instrumentation & Measurement, 2023

## Journals:
- IEEE Robotics and Automation Letters
- IEEE Transactions on Instrumentation & Measurement, 43 Papers
- IEEE Sensors
- IEEE Instrumentation & Measurement Magazine, 1 Paper
- IEEE Open Access Journal on Circuits and Systems, 1 Paper
- Wiley Journal of Field Robotics
- Elsevier Automatica
- Elsevier Aerospace Science and Technology, 15 Papers
- Elsevier Measurement, 6 Papers
- Springer Visual Computing for Industry, Biomedicine, and Art
- Space: Science & Technology, 4 Papers
- The Aeronautical Journal, 3 Papers

## Conferences:
- International Conference on Robotics and Automation (ICRA) 2024
- International Conference on Learning Representations (ICLR) 2024
- International Federation of Automatic Control (IFAC) World Congress 2023, 1 Paper
- American Control Conference (ACC) 2023, 2024

---

# Leadership Experience

## Space Generation Advisory Council
*Nov 2020 to Present*

*Role: Mentor for AI and Space Technology*

**Mentorship & Guidance**:
- Provided personalized mentorship to **10+ professionals** in computer vision and satellite data processing
- Offered targeted guidance and sustained support to mentees, fostering future leaders in space technology
- Conducted one-on-one sessions covering career development, research methodology, and technical skills

**Curriculum Development**:
- Designed comprehensive curriculum covering:
  - Deep learning fundamentals and advanced architectures
  - Computer vision for satellite imagery analysis
  - AI applications in space exploration and robotics
  - Practical implementation with PyTorch and TensorFlow
- Created structured learning pathways for professionals at different skill levels

**Workshops & Training**:
- Led **multiple workshops** on AI applications in space technology
- Topics covered: satellite image segmentation, object detection in orbital imagery, autonomous spacecraft navigation
- Created extensive training materials including:
  - Tutorial notebooks for satellite image analysis
  - Documentation on implementing CV algorithms for space data
  - Best practices for deploying ML models in space applications

**Community Impact**:
- Contributing to capacity building in the space technology sector
- Fostering interdisciplinary collaboration between AI and space engineering communities

## World Astronomy Week (Iran)
*Jan 2017 to Jan 2023*

*Role: Executive Member*

## Astronomy Outreach, University of Tehran
*Jan 2021 to Sep 2023*

*Role: Executive Member*

## Iran Martial Arts Federation
*Mar 2016 to Present*

*Role: Martial Arts Instructor*

- Instructed 400+ from diverse backgrounds, cultivating essential communication and life skills while fostering physical and mental well-being
- Nurtured discipline, leadership, and teamwork within the context of martial arts for comprehensive personal development
- Black Belt Dan II

---

# Honors, Prizes, Awards, and Fellowships

## AI + Physics
*IAIFI + MIT - Jul 2024*

## MLx Representation Learning & Generative AI
*Oxford Machine Learning Summer School - Jun 2024*

## Outstanding Reviewer
*IEEE Transactions on Instrumentation & Measurement - 2023*

Recognized for exceptional contributions to the peer review process

## AWS AI & ML Scholarship 2023
*$4000 Valued*

**Endowed by:** Amazon and Udacity

**Criteria:** Completing ML course, implementing Reinforcement Learning for optimal agent decision-making and achieving a sub-one-minute lap in AWS DeepRacer Student League. Awarded to top 1% of applicants.

## Ranked 1st in OxML Competition Track
*2023*

**Criteria:** Attained the highest accuracy (82.3%) in cancer cell detection among 120+ international participants

## Top 10% in M.Sc. Aerospace Engineering
*2019*

**Criteria:** Ranked in the top 10% among 5000+ participants in the National University Entrance Exam

## Ranked 1st in Class 2019
*University of Tehran, College of Interdisciplinary Science and Technology - 2021*

**Criteria:** Attained the highest GPA (4.0 out of 4.0) in the class

## Medalist, Iran Martial Arts Federation, National Competitions
- **Gold Medalist:** 2011, 2012, 2018, and 2019
- **Silver Medalist:** 2015
- **Bronze Medalist:** 2016 and 2019

## Black Belt Dan II
*Nearu Martial Arts, 2015*

Achieved due to advanced mastery in martial arts techniques and philosophy, requiring a decade of dedicated training, highlights exceptional skill, knowledge, and commitment
