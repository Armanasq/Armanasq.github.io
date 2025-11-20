---
layout: archive
title: "CV"
url: /cv/
redirect_from:
  - /resume
---


Education
======
* M.Sc. in Space Engineering, University of Tehran, 2022 **(GPA:4.0/4.0)**
  * School Ranking: 1st in Iranian Universities (U.S, News)
  * Oxford Machine Learning Summer School (OxML 2022)
  * Thesis: AI Application in Inertial Navigation
    * Sensor fusion algorithms were combined with Deep Learning to improve inertial attitude estimation accuracy
    * Ray and Sherpa were used for Hyperparameter Optimization (PBT, Grid & Random Search) in Python (Keras & Pytorch)
    * End-to-End ANN Frameworks were developed for Inertial Odometry (6DoF & 9DoF)
    * End-to-End ANN Frameworks were developed for Attitude Estimation (2DoF & 3DoF)
* MBA in Healthcare, Academic Center for Education, Culture and Research, 2020 **(GPA:4.0/4.0)**
  * Projects:
    * Elderly tourism
    * Application of AI in personalized medicine
* B. Eng. in Avionics, Aviation Industry Training Center, 2019 **(GPA:3.8/4.0)**
* A.E.T in Avionics, Civil Aviation Technology College, 2016

Professional Experience
======
* **Lead Machine Learning Engineer**, AI-Ark - **Dec. 2024 – Oct. 2025**
  * Architected hybrid vector search system with Nomic-Embed-1.5B and Vespa/Elasticsearch
  * Optimized ANN/KNN ranking profiles improving F1-score from 0.76 to 0.92
  * Implemented 4-bit and 8-bit quantization using GGUF and AWQ techniques
  * Led development of multi-turn RAG chatbot with persistent memory
  * Led team of 6 engineers in financial behavioral analysis systems achieving 95.2% accuracy

* **Lead AI Engineer**, Fanaavaran Farayand Farda - **Jun. 2024 – Dec. 2024**
  * Applied few-shot learning to train detection/segmentation model on 53 classes achieving 81% mAP
  * Reduced model latency by 70% through INT8 quantization and TensorRT conversion
  * Deployed models via TorchServe supporting 200+ concurrent requests
  * Built document intelligence platform reducing processing time from 2 mins to 30 secs

* **AI Researcher (Part-time)**, Fasta Robotics - **Nov. 2023 – Oct. 2025**
  * Implemented ORB-SLAM3 visual-inertial mapping system on ROS2
  * Developed VIO-LiDAR sensor fusion achieving 15cm localization accuracy
  * Deployed YOLOv12 object detection achieving 95% pedestrian detection accuracy
  * Designed end-to-end learning framework improving robustness by 30%

* **Machine Learning Engineer**, Rajaei Cardiovascular Center - **May 2022 – Jun. 2024**
  * Built search architecture achieving 92% precision across PubMed/Scopus databases
  * Architected RAG system for 100K+ clinical documents
  * Developed clinical decision framework with 87% diagnostic accuracy
  * Built automated ICD-10 coding system with 91% accuracy
  * Managed team of 5 ML engineers

* **AI Research Scientist**, Farzan Research Institute - **Oct. 2021 – Jun. 2024**
  * Fine-tuned Nemotron, AYA-101 and BERT models using LoRA
  * Optimized training on A100/H100 GPUs using DeepSpeed ZeRO-3
  * Engineered hybrid retrieval systems (BM25, FAISS, cross-encoders)
  * Implemented QLoRA 4-bit quantization on A100 clusters

* **Computer Vision Engineer & Freelancer** - **Apr. 2020 – Oct. 2021**
  * Implemented LSTM image captioning achieving BLEU score of 0.32
  * Built CNN-LSTM handwriting recognition with 5.2% CER
  * Created Mask R-CNN segmentation achieving 43.5 mAP on COCO dataset
  * Developed PyTorch training pipelines with distributed computing

* **Martial Arts Instructor**, Iran Martial Arts Federation - **Mar. 2016 – Present**
  * Black Belt Dan II
  * Instructed 400+ students from diverse backgrounds

Technical Expertise
======
* **Model Architectures**
  * Transformers, CNNs (YOLO, ResNet), RNNs (LSTM, GRU)
* **LLMs & NLP**
  * PEFT/LoRA fine-tuning, 4-bit & 8-bit quantization (GGUF, AWQ)
  * Prompt engineering, RAG systems
  * HuggingFace, Unsloth, AWS Bedrock, Azure AI
* **Computer Vision**
  * Object Detection, Segmentation
  * Vision Language Modeling
  * YOLO family, Mask R-CNN, Vision Transformers
* **Autonomous Systems**
  * ROS2, LiDAR-SLAM, ORB-SLAM3
  * State Estimation, Multi-Sensor Fusion
  * Visual-Inertial Odometry
* **Data Infrastructure**
  * Vector databases (FAISS, Qdrant, Vespa, Elasticsearch)
  * BM25/TF-IDF indexing
  * Hybrid retrieval systems
* **ML Frameworks**
  * PyTorch, TensorFlow, Keras
  * Docker, FastAPI, TorchServe
  * DeepSpeed, mixed precision training
* **Programming**
  * Python, C++, CUDA, Git, LaTeX

Publications
======
  <ul>
  {{ range .Site.RegularPages.ByType "publication" }}
    <h2>{{ .Title }}</h2>
    <!-- Add any other desired content or template partials here -->
{{ end }}
</ul>
  

  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Research Experience
======
* **Physics-Informed AI Researcher**, Students' Scientific Research Center - **Jul. 2024 – Sep. 2024**
  * Built TE-PINN: transformer-physics network for IMU orientation estimation
  * Designed multi-head attention for IMU data, reduced error 36.8%
  * Embedded quaternion kinematics and rigid body dynamics
  * Research selected for ICRA 2025; code and datasets publicly available

* **Research Participant**, MIT IAIFI AI+Physics Program - **May 2024 – Jun. 2024**
  * Developed Vision Transformer-based tumor severity simulator
  * Applied Sequential Neural Posterior Estimation (SNPE)
  * Achieved 87% diagnostic accuracy on medical imaging datasets
  * Presented at cross-institutional workshop (75+ participants from MIT, Harvard)

* **Competition Winner**, Oxford Machine Learning Summer School - **May 2023 – Aug. 2023**
  * Implemented transfer learning achieving 82% accuracy
  * Ranked 1st among 120+ participants in international competition
  * Developed ensemble approach combining EfficientNet and Vision Transformer models

* **Graduate Researcher**, University of Tehran - **Sep. 2019 – Sep. 2022**
  * Developed end-to-end learning models reducing attitude estimation error by 40%
  * Evaluated on 7 datasets (100+ km IMU measurements)
  * Published in Measurements journal (Elsevier, IF 5.2)
  * Open-sourced implementation [+33 GitHub Stars]

* **Referee of Research Council**, Students' Scientific Research Center - **Apr. 2019 – Present**
  * Evaluated research proposals for medical science and healthcare solutions

Lab Experience 
======
* Space Lab, University of Tehran - **Sep. 2019 – Sep. 2022**
  * Used 3-DoF experimental test bed for integrated attitude dynamics and control
  * Used LabView and ARM development boards
* Fuzzy Logic Lab, University of Tehran, and USERN - **Nov. 2019 – Present**
  * Done research on Fuzzy Inference Systems based projects, such as Fuzzy tuned complementary filters for IMU-based attitude estimation. Advisor: Dr. M. H. Sabour
  * Websites
    * [University of Tehran Fuzzy Logic Lab](https://fuzzylogic.ut.ac.ir/en/page/7122/arman-asgharpoor)
    * [USERN Fuzzy Logic Lab Interest Group](https://usern.tums.ac.ir/Group/Info/FLLIG)
    * [ResearchGate](https://www.researchgate.net/lab/Fuzzy-Lab-Mohammad-H-Sabour)
* Avionics Lab, Aviation Industry Training Center - **Sep. 2018 – Sep. 2020**
  * Worked with Aircraft Instrument Panels and different flight instruments
    * Altimeter, Attitude, Airspeed, Vertical Speed, and Heading Indicator
* Electronics Lab, Aviation Industry Training Center - **Sep. 2018 – Sep. 2020**
  * Designed and assembled PCBs: Fire extinguisher, Flight Management System (FMS) simulator and etc.
  * Used various measuring tools: Function Generator, Oscilloscope, LCR meter, etc.
* Aircraft Instruments Lab, Civil Aviation Technology College -  **Oct. 2015 – Aug. 2016**
  * Lab redesigned, and inventory management was done to improve student performance
  * Repaired different flight instruments

Leadership Experience
======
* Universal Scientific Education & Research Network - **Jan. 2021 – Jan. 2022**
  * 6th International USERN Congress & Prize Awarding Festival, Executive Member
  * Research Week, Executive Member
  * Lab Techniques School, Executive Member
  * Minatare Talk, Executive Member
  * USERN Health & Art, 7th International Festival of Paintings for Pediatric Patients, Executive Member
  * R&D, Publicity, Media, and IT, Team Member
* Tehran University of Medical Sciences - **Jan. 2014 – Dec. 2021**
  * 4th Student Education Development Festival, Executive Member
  * 20th, 21st, and 23rd Conference of Annual General Meeting, Executive Member
* University of Tehran - **Sep. 2019 – Jul. 2020**
  * Cultural Society KARA, Former
  * Climate Change Conference, Organizer
* Night Sky Institute - **Mar. 2017**  
  * World Astronomy Week, Executive Member
* Civil Aviation Technology College - **Jan. 2012 – Sep. 2015**
  * WaterRocket Competition, Organizer
  * Road & Urban Development & The Related Industries Exhibition, Executive Member
  * 3rd, 4th, and 5th International Aviation & Space Industries Exhibition of Iran, Executive Member
* Pest Control – Volunteer Work - **Jan.2013 – Present**
  * Kahrizak Nursing Home
  * Sarai Ehsan Social Victims Center
  * Vardavard Welfare

Certifications
======
* **AI + Physics**, IAIFI + MIT - **Jul. 2024**
* **MLx Representation Learning & Generative AI**, Oxford Machine Learning Summer School - **Jun. 2024**
* **AI for Global Goals**, Oxford Machine Learning Summer School - **Jun. 2023**
* **Deep Learning for Computer Vision**, University of Colorado Boulder (Coursera) - **Jun. 2023**
* **AI Programming with Python**, Udacity - **Jan. 2024**
* **State Estimation and Localization for Self-Driving Cars**, University of Toronto (Coursera) - **Feb. 2023**
* **Spacecraft Dynamics and Control Specialization**, University of Colorado Boulder (Coursera) - **Aug. 2020**
  * Kinematics: Describing the Motions of Spacecraft
  * Spacecraft Dynamics Capstone: Mars Mission
  * Control of Nonlinear Spacecraft Attitude Motion
  * Kinetics: Studying Spacecraft Motion
* **Neural Networks and Deep Learning**, DeepLearning.AI (Coursera) - **May 2023**
* **MATLAB Onramp**, MathWorks
* USERN:
  * Submission & Peer Reviewing
  * Data Analysis in SPSS
  * Systematic Review
  * Scientific Writing
  * Meta-analysis

Membership & Affiliations
======
* University of Oxford Responsible Technology Institute Student Network
* University of Tehran Chess Club
* Student Chapter of the European Low Gravity Research Association
* Astronomers Without Borders
* Space & Satellite Professionals International
* Royal Aeronautical Society
* Educational Development Center, TUMS
* Space Generation Advisory Council

Awards & Honors
======
* **Outstanding Reviewer**, IEEE Transactions on Instrumentation & Measurement - **2023**
* **AWS AI & ML Scholarship** (awarded to top 1% of applicants) - **2023**
  * $4000 Valued, Endowed by Amazon and Udacity
* **Ranked 1st in OxML Competition Track** (120+ international participants) - **2023**
  * Achieved 82.3% accuracy in cancer cell detection
* **Ranked top 10% in M.Sc. Aerospace Engineering** National University Entrance Exam - **2019**
  * Among 5000+ participants
* **Ranked 1st in Class 2019**, University of Tehran, College of Interdisciplinary Science and Technology - **2021**
  * GPA: 4.0/4.0
* **Iran Martial Arts Federation**, National Competitions
  * Gold Medalist - **2011, 2012, 2018, 2019**
  * Silver Medalist - **2015**
  * Bronze Medalist - **2016, 2019**
* **Black Belt Dan II**, Nearu Martial Arts - **2015**
  * Requiring a decade of dedicated training
* **USERN Miniature Talk**, Appreciated Presenter - **Aug. 2021**