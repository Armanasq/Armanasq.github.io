---
title: "Training Your Own BERT Model from Scratch "
date: 2023-08-31
description: "Step-by-Step Tutorial on Training Your Own BERT Model"
url: "/nlp/train-BERT/"
showToc: true
math: true
disableAnchoredHeadings: false
commentable: true
tags:
  - Natural Language Processing
  - NLP
  - Tutorial
  - BERT
---
[&lArr; Natural Language Processing](/nlp/)
<img src="/bert.png" alt="BERT" style="width:100%;display: block;
  margin-left: auto;
  margin-right: auto; margin-top:0px auto" >
</div>



# Training Your Own BERT Model from Scratch 🚀

Hey there, fellow learner! 🤓 In this post, we're going to embark on an exciting journey to train your very own BERT (Bidirectional Encoder Representations from Transformers) model from scratch. BERT is a transformer-based model that has revolutionized the field of natural language processing (NLP). Most of current tutorial only focus on fine-tuning the existing pre-trained model. By the end of this tutorial, you'll not only understand the code but also the intricate details of the methodologies involved.

---

## Section 0: Introduction 🚀

BERT has revolutionized the field of NLP by offering pre-trained models that capture rich contextual information from large text corpora. However, training a BERT model tailored to specific tasks or languages requires careful consideration and meticulous steps.

### The Power of Custom BERT Models

Custom BERT models empower researchers, data scientists, and developers to harness the capabilities of BERT while fine-tuning it for unique use cases. Whether you're working on a specialized NLP task, dealing with languages with complex structures, or tackling domain-specific challenges, a custom BERT model can be your ally.

### What We Will Cover

Throughout this tutorial, we will delve into every aspect of creating and training a custom BERT model:

- **Data Preparation**: We'll start by preparing a diverse and substantial text corpus for training. 

- **Tokenization**: We'll explore tokenization, the process of breaking down text into smaller units for analysis. BERT relies on subword tokenization, a technique that can handle the complexity of various languages and word structures.

- **Model Tokenizer Initialization**: We'll initialize the model tokenizer, which is responsible for encoding text into input features that our BERT model can understand. 

- **Preparing Data for Training**: We'll dive into the crucial steps of preparing our text data for BERT training. This includes masking tokens for masked language modeling (MLM), one of BERT's key features.

- **Creating a Custom Dataset**: We'll construct a custom PyTorch dataset to efficiently organize and load our training data. 

- **Training Configuration**: We'll configure our BERT model, specifying its architecture and parameters. These configurations define the model's behavior during training.

- **Model Initialization**: We'll initialize the BERT model for MLM, ensuring that it's ready to learn from our data. This step includes handling GPU placement for accelerated training.

- **Training Loop and Optimization**: We'll set up the training loop and optimize our model using the AdamW optimizer. 

- **Testing Your Custom BERT Model**: Finally, we'll put our trained model to the test with real-world examples. 

Let's begin our journey by delving into the critical step of data preparation.

---
## Section 1: Prerequisites and Setup 🛠️

Before we dive into the BERT training process, let's ensure you have the necessary tools and libraries installed. We'll be using Python, so make sure you have it installed on your system.

### Setting up the Environment

```python
# Uninstall any existing TensorFlow versions (if applicable)
!pip uninstall -y tensorflow

# Install the 'transformers' library from the Hugging Face repository
!pip install git+https://github.com/huggingface/transformers

# Check the installed versions
!pip list | grep -E 'transformers|tokenizers'
```

We'll also need your Hugging Face API token (`token`) for later steps, so make sure you have it ready.

```python
# Import the relevant libraries for logging in
from huggingface_hub import login
login(token=`your_huggingface_token`)
```

---

## Section 2: Data Preparation 📊


### The Significance of Data

The richness, diversity, and volume of our data directly impact the model's language understanding and generalization capabilities. Our training data must be representative of the language and tasks our model will encounter in the real world which ensures that our model learns relevant patterns and information. 

A diverse dataset exposes the model to various language styles, topics, and domains to enhances the model's ability to handle a wide range of inputs. 

The size of the dataset matters. Larger datasets enable the model to learn more robust language representations, but they also require substantial computational resources.

### Data Collection and Sources

When collecting data for training a Language model, consider various sources such as books, articles, websites, and domain-specific texts. Open-source datasets and publicly available corpora can be valuable resources. In this tutorial, we obtained data from the OSCAR project (Open Super-large Crawled Aggregated coRpus) which is an Open Source project aiming to provide web-based multilingual resources and datasets. You can find the project page [here](https://oscar-project.org/). Also, you can use any other text data you have access.

### Data Preprocessing

Data preprocessing involves tasks like text cleaning, sentence tokenization, and ensuring uniform encoding (e.g., UTF-8). Proper preprocessing minimizes noise and ensures that the text is ready for tokenization.

### Corpus Size and Sampling

The size of your corpus can vary based on your resources, but more data is generally better. If working with a massive dataset isn't feasible, consider random sampling or domain-specific sampling to create a representative subset.

### Domain-Specific Considerations

If your NLP task is domain-specific (e.g., medical or legal text), focus on collecting data relevant to that domain. Domain-specific terminology and context are crucial for accurate model performance.


Let's start by loading the dataset and setting up data loading workers. The OSCAR dataset support various languages, including Persian (Farsi). We used the HuggingFace framework to load the OSCAR dataset via `datasets` library.

```python
# Install the 'datasets' library
!pip install datasets

# Load the streaming dataset
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset('oscar', 'unshuffled_deduplicated_fa', streaming=True)

# Create a DataLoader with multiple workers to parallelize data loading
dataloader = DataLoader(dataset, num_workers=20)

# Access the training data
training_data = dataset['train']

# Explore dataset features
features = training_data.features
```

---

## Section 3: Tokenization 🧩

Tokenization is the process of breaking down text into smaller units, called tokens, and is a fundamental step in natural language processing. In this section, we'll explore tokenization in depth, especially subword tokenization, which is pivotal for BERT models.

### The Role of Tokenization

Tokenization is akin to breaking a sentence into individual words or subword units. It serves several critical purposes in NLP and BERT training:

#### 1. Text Segmentation:
Tokenization segments text into units that are easier to process. These units can be words, subwords, or even characters.

#### 2. Vocabulary Building:
Tokenization contributes to building a vocabulary of unique tokens. This vocabulary is essential for encoding and decoding text.

#### 3. Consistency:
Tokenization ensures consistency in how text is represented. The same word or subword is tokenized consistently across different documents.

#### 4. Handling Variations:
Tokenization handles variations like verb conjugations, pluralization, and capitalization, ensuring that similar words map to the same tokens.

### Subword Tokenization

BERT, and many other state-of-the-art models, rely on subword tokenization rather than word-based tokenization. Subword tokenization is a more flexible and effective approach, especially for languages with complex word structures.

#### Byte-Pair Encoding (BPE)

One popular subword tokenization technique is Byte-Pair Encoding (BPE). BPE divides text into subword units, such as prefixes, suffixes, and root words. This approach can represent a wide range of words and word variations effectively.

#### Vocabulary Size

When using BPE tokenization, you specify the vocabulary size, which determines the number of unique subword tokens. A larger vocabulary can capture more fine-grained language patterns but requires more memory.

#### Special Tokens

BERT models use special tokens like:

- '[CLS]' (classification)
- '[SEP]' (separator)
- '[MASK]' (masked)

These tokens have specific roles in model input.

### Tokenization with Hugging Face Transformers

The Transformers library from Hugging Face provides powerful tools for tokenization. You'll use these tools to encode text into input features that the BERT model can understand.

Tokenization is the bridge between raw text data and the model's input format. A strong understanding of tokenization is crucial for interpreting how the model processes and understands language.

```python
# Define the directory to save tokenization files
output_directory = './data/text/oscar_fa/'

# Create the output directory if it doesn't exist
import os
os.makedirs(output_directory, exist_ok=True)

# Tokenization Process
text_data = []
file_count = 0

for sample in training_data:
    sample_text = sample['text'].replace('\n', '')
    text_data.append(sample_text)
    
    if len(text_data) == 10_000:
        # Save tokens to a file every 10,000 samples
        with open(os.path.join(output_directory, f'text_{file_count}.txt'), 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

# Save any remaining tokens
if text_data:
    with open(os.path.join(output_directory, f'text_{file_count}.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))
```

Now that we have tokenized our data, let's train a Byte-Level Byte-Pair-Encoding (BPE) tokenizer. Tokenizers like this split words into subword units, making them suitable for a wide range of languages.

```python
# Get file paths of tokenization files
from pathlib import Path
tokenization_files = [str(x) for x in Path('./data/text/oscar_fa').glob('**/*.txt')]

# Initialize and train the tokenizer
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=tokenization_files[:5], vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

# Create a directory for the tokenizer model
os.mkdir('./filiberto')

# Save the trained tokenizer model
tokenizer.save_model('filiberto')
```

---

## Section 4: Model Tokenizer Initialization 🤖🔡

Now that we understand the importance of tokenization, let's explore how to initialize the model tokenizer. The tokenizer plays a critical role in encoding text into input features that our BERT model can comprehend. In this section, we'll dive into the methodology behind tokenizer initialization.

### Tokenizer Initialization

The model tokenizer is a vital component of the BERT architecture. It's responsible for several key tasks:

#### 1. Tokenization:
The tokenizer breaks down input text into tokens, including subword units and special tokens like '[CLS]' and '[SEP]'.

#### 2. Vocabulary Handling:
It manages the model's vocabulary, which includes all the unique tokens the model understands. The vocabulary is a crucial part of encoding and decoding text.

#### 3. Encoding:
The tokenizer encodes text into input features, such as input IDs and attention masks, which are used as inputs for the model.

### Hugging Face Transformers for Tokenization

Hugging Face's Transformers library simplifies tokenization with a user-friendly interface. Here's a step-by-step breakdown of initializing the model tokenizer:

#### 1. Pretrained Models:
Transformers provides a wide range of pretrained BERT models for various languages and tasks. You can choose a model that matches your requirements.

#### 2. Tokenizer Loading:
Load the model's tokenizer using the selected pretrained model's name. This ensures that the tokenizer is aligned with the model's architecture.

#### 3. Tokenization:
You can now use the tokenizer to tokenize any text input. It returns tokenized inputs, including input IDs and attention masks.

#### 4. Special Tokens:
The tokenizer automatically handles special tokens like '[CLS]', '[SEP]', and '[MASK]' according to the model's specifications.

Understanding the tokenizer's role and its initialization is crucial for effectively preparing text data for BERT training. Tokenization transforms raw text into a format that the model can process, setting the stage for the subsequent steps in model training.

---

## Section 5: Preparing Data for Training 📝

Having grasped the significance of tokenization and model tokenizer initialization, let's delve into the process of preparing our text data for BERT training. This section outlines the steps to create input features for our model, including masked language modeling (MLM).

### Preparing Data for Training

Training a BERT model involves creating input data that suits the model's architecture and objectives. BERT is pretrained using a masked language modeling (MLM) task, which involves predicting masked words in a sentence. To adapt our data for this task, we follow these steps:

#### 1. Tokenization:
We use the model tokenizer to tokenize our text data, resulting in input IDs and attention masks.

#### 2. MLM Masking:
To create an MLM task, we randomly mask a percentage (usually 15%) of the tokens in the input IDs. We exclude special tokens like '[CLS]' and '[SEP]' from masking.

#### 3. Label Preparation:
The original token IDs are retained as labels for the model. During training, the model learns to predict the original tokens from the masked inputs.

By preparing our data in this manner, we set up a supervised learning task for our BERT model. It learns to understand the context of a sentence by predicting masked words, which is a fundamental part of BERT's bidirectional learning.

The concept of MLM is central to how BERT learns rich language representations. By predicting masked tokens, the model gains an understanding of how words relate to each other within sentences.

In our next section, we'll explore how to construct a custom PyTorch dataset to efficiently organize and load our prepared training data, another crucial aspect of BERT model training.

---

## Section 6: Creating a Custom Dataset 📚

Training a BERT model requires organizing and loading the prepared data efficiently. In this section, we'll explore how to create a custom PyTorch dataset to accomplish this task.

### The Role of Datasets

Datasets are the backbone of deep learning. They allow us to efficiently organize and load data, making it ready for training. A custom dataset class is particularly useful for preparing data in a format that's compatible with PyTorch's data loading utilities.

### The Custom Dataset Class

To create a custom PyTorch dataset, we define a class that inherits from the `torch.utils.data.Dataset` class. Our custom dataset class will have the following key methods:

#### 1. Initialization (`__init__`):
In the constructor, we set up any necessary data structures and configurations. Here, we pass in the prepared encodings of our text data.

#### 2. Length (`__len__`):
The `__len__` method returns the number of samples in the dataset. For BERT training, this corresponds to the number of input sentences.

#### 3. Get Item (`__getitem__`):
The `__getitem__` method retrieves a specific sample from the dataset. It returns a dictionary containing the input features, attention masks, and labels for that sample.

This custom dataset class efficiently encapsulates our data and makes it compatible with PyTorch's data loading utilities.

### Efficient Data Loading

Using a custom dataset class, we can leverage PyTorch's `DataLoader` to efficiently load and iterate through our data during training. This setup ensures that our BERT model receives properly formatted data in batches, facilitating the training process.

Custom datasets are a fundamental concept in deep learning, and understanding how to create and use them is essential for efficient data handling in PyTorch. In the next section, we'll delve into the crucial aspects of configuring our BERT model for training.

---

## Section 7: Training Configuration 🚀

With our data prepared and organized, it's time to configure our BERT model for training. In this section, we'll explore the key configuration settings that influence the training process.

### Model Configuration

Before we can embark on training our BERT model, we need to specify its architecture and behavior. The 'config' object contains various parameters that define how the model will operate during training. Let's dive into some of the crucial settings:

#### 1. Vocabulary Size (`vocab_size`):
The 'vocab_size' parameter determines the size of the model's vocabulary. It's crucial to match this value with the vocabulary size used during tokenization to ensure compatibility.

#### 2. Maximum Position Embeddings (`max_position_embeddings`):
This parameter sets the maximum number of positions the model can handle. In practice, this value should be set to match the maximum sequence length used during tokenization.

#### 3. Hidden Size (`hidden_size`):
'hidden_size' specifies the dimensionality of the model's hidden states. This parameter plays a crucial role in determining the model's capacity to capture complex patterns in the data.

#### 4. Number of Attention Heads (`num_attention_heads`):
'num_attention_heads' controls the number of attention heads in the

 model's multi-head attention mechanism. Increasing this value can enhance the model's ability to capture fine-grained relationships in the data.

#### 5. Number of Hidden Layers (`num_hidden_layers`):
'num_hidden_layers' defines how deep the model's architecture is. Deeper models can capture more complex patterns but require more computational resources.

#### 6. Type Vocabulary Size (`type_vocab_size`):
'type_vocab_size' is typically set to 1 for tasks like masked language modeling.

By configuring these parameters, we define the architecture and behavior of our BERT model. These settings can be adjusted based on the specific requirements of your NLP task and the available computational resources.

### Model Initialization

Now that we've defined our model's configuration, it's time to initialize the BERT model for masked language modeling (MLM). We'll also handle device placement to leverage GPU acceleration if available.

#### Model Selection:
We initialize a BERT model for masked language modeling (MLM) using the 'RobertaForMaskedLM' class from the Transformers library. This class provides a pre-configured BERT architecture ready for fine-tuning.

#### Device Placement:
We check if a GPU is available, and if so, we move the model to the GPU using `model.to(device)`. This step is essential for leveraging GPU acceleration during training, which significantly speeds up the process.

Initializing the model is a pivotal step in our BERT training journey. With the model in place, we're ready to dive into the training loop and optimize it for our specific task.

---

## Section 8: Training Loop and Optimization 🔄

Let's dive into the heart of training. We'll set up the training loop and optimize our model using AdamW.

### Activation and Optimization

#### Training Mode:
We activate the training mode of our BERT model using `model.train()`. This step is essential because it tells the model to compute gradients and update its parameters during training.

#### Optimizer Initialization:
We initialize an AdamW optimizer, a variant of the Adam optimizer designed for training deep learning models. The optimizer is responsible for adjusting the model's weights to minimize the loss.

#### Training Epochs:
We specify the number of training epochs. An epoch represents one complete pass through the training data. In this example, we've set it to 2, but you can adjust this based on your specific training needs.

### Training Loop

#### TQDM Progress Bar:
We use the TQDM library to create a progress bar that tracks the training progress. This progress bar provides real-time feedback on training loss and completion.

#### Batch Processing:
Inside the training loop, we process data in batches. Each batch consists of input IDs, attention masks, and labels.

#### Forward Pass:
We perform a forward pass through the model, passing the input tensors and computing model predictions.

#### Loss Calculation:
We extract the loss from the model's outputs. The loss represents how far off the model's predictions are from the ground truth labels.

#### Backpropagation:
We perform backpropagation to calculate gradients with respect to the model's parameters. These gradients guide the optimizer in adjusting the model's weights to minimize the loss.

#### Parameter Update:
We update the model's parameters using the optimizer's `step` method. This step is where the model learns from its mistakes and becomes better at its task.

#### Progress Bar Updates:
We update the progress bar to display the current epoch and the loss for the current batch.

### Model Saving

Finally, we save the pre-trained model using `model.save_pretrained('./filiberto')`. It's crucial to save the model's weights and configuration after training so that you can later load and use the trained model for various NLP tasks.

This section provides a comprehensive overview of the training process, from setting up the training loop to optimizing our BERT model for the specified task.

---

## Section 9: Testing Your Custom BERT Model 🧪

Congratulations! 🎉 You've trained your custom BERT model. Let's test it with a fun example.

### Model Inference

#### Model Loading:
We load the trained model for inference using the Transformers library. The `pipeline` function simplifies the process of using the model for specific tasks, in this case, masked language modeling.

#### Testing with Masked Sentence:
We create a masked sentence to test our model's ability to fill in the missing word. The '[MASK]' token in the sentence indicates the position where the model should make predictions.

#### Inference Results:
We run the masked sentence through the model using the `fill` pipeline and obtain the model's predictions. These predictions reveal the model's language understanding and completion capabilities.

Testing your trained BERT model on real-world examples is an essential step in evaluating its performance and ensuring that it can effectively handle language-related tasks.

---

## Conclusion 🎓

You've reached the end of this comprehensive tutorial on training a custom BERT model from scratch using the Transformers library from Hugging Face. Throughout this tutorial, we've covered essential topics, including data preparation, tokenization, model configuration, training, and inference.

Training your own BERT model is a powerful capability that allows you to fine-tune a language model for specific tasks and domains. It's important to note that training large models like BERT requires significant computational resources, so you may want to consider using pre-trained models and fine-tuning them for your specific needs if you have limited resources.

As you continue to explore the world of natural language processing (NLP) and deep learning, this tutorial should serve as a valuable foundation for creating and customizing state-of-the-art language models. Remember to adapt the techniques and configurations discussed here to your specific tasks and datasets, and happy modeling! 🤖📚🚀