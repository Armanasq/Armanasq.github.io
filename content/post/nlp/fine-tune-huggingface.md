---
title: "Fine Tune Pre-Trained Models with Hugging Face"
date: 2023-08-31
description: "Comprehensive Tutorial on Fine Tuning Pre-Trained Models with Hugging Face"
url: "/nlp/fine-tune-hugging-face/"
showToc: true
math: true
disableAnchoredHeadings: false
commentable: true
tags:
  - Natural Language Processing
  - NLP
  - Tutorial
---
[&lArr; Natural Language Processing](/nlp/)
<img src="/hf-logo-with-title.png" alt="ROS" style="width:100%;display: block;
  margin-left: auto;
  margin-right: auto; margin-top:0px auto" >
</div>



# Fine Tune Pre-Trained Models with Hugging Face

In the realm of Natural Language Processing (NLP), harnessing the capabilities of pre-trained models is a fundamental endeavor. These models, having already learned from vast amounts of text data, serve as valuable starting points for various NLP tasks. Hugging Face, a library widely embraced by NLP practitioners, simplifies the process of fine-tuning pre-trained models to suit specific applications.

Fine-tuning entails refining a pre-trained model by training it further on domain-specific data. This process allows the model to adapt and excel in tasks like text classification, sentiment analysis, and more. With its user-friendly interfaces and comprehensive functionalities, Hugging Face empowers us to effectively fine-tune these models, bridging the gap between general language understanding and specific tasks.

This tutorial delves into the pragmatic intricacies of fine-tuning pre-trained models using Hugging Face. We will explore the nuances of preparing datasets, navigating hyperparameters, visualizing training progress, and employing advanced training techniques. By the end of this tutorial, you'll be well-equipped to unlock the full potential of pre-trained models for your NLP undertakings.

## Section 1: Prepare Your Dataset

### Data Exploration

Before delving into preprocessing, it's essential to understand your dataset's structure. For instance, let's explore the Yelp Reviews dataset, which consists of reviews and associated ratings. Understanding your data is crucial, so let's begin by gaining insights into the dataset's dimensions:

```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

# Display the number of training and test examples
print("Number of training examples:", len(dataset["train"]))
print("Number of test examples:", len(dataset["test"]))

# Print a few training examples
for i in range(3):
    print("Example", i, ":", dataset["train"][i]["text"])
    print("Rating:", dataset["train"][i]["label"])
    print("----------")
```

```shell
Number of training examples: 650000
Number of test examples: 50000
Example 0 : dr. goldberg offers everything i look for in a general practitioner.  he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first.  really, what more do you need?  i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank.
Rating: 4
----------
Example 1 : Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff.  It seems that his staff simply never answers the phone.  It usually takes 2 hours of repeated calling to get an answer.  Who has time for that or wants to deal with it?  I have run into this problem with many other doctors and I just don't get it.  You have office workers, you have patients with medical needs, why isn't anyone answering the phone?  It's incomprehensible and not work the aggravation.  It's with regret that I feel that I have to give Dr. Goldberg 2 stars.
Rating: 1
----------
Example 2 : Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years and is really all about the big picture. It is because of him, not my now former gyn Dr. Markoff, that I found out I have fibroids. He explores all options with you and is very patient and understanding. He doesn't judge and asks all the right questions. Very thorough and wants to be kept in the loop on every aspect of your medical health and your life.
Rating: 3
----------
```

This exploration gives you a bird's-eye view of your dataset, aiding in identifying potential imbalances and understanding its distribution.

### Tokenization Strategies

Tokenization lies at the core of NLP tasks, where text input is transformed into a sequence of tokens understandable by machine learning models. Hugging Face provides a versatile Tokenizer class that simplifies this crucial step. Tokenizers are available for various models and come in two flavors: a full Python implementation and a "Fast" implementation based on the Rust library, which offers improved performance, especially for batched tokenization.

#### Preparing Inputs

The main classes for tokenization are `PreTrainedTokenizer` and `PreTrainedTokenizerFast`, which are base classes for all tokenizers. These classes provide common methods for encoding string inputs into model-ready inputs. They handle tokenizing (splitting text into sub-word token strings), converting token strings to IDs and vice versa, and encoding/decoding (tokenizing and converting to integers). Additionally, they facilitate the management of special tokens like mask, beginning-of-sentence, etc., ensuring they are not split during tokenization.

#### Special Tokens

Hugging Face tokenizers offer attributes for important special tokens like `[MASK]`, `[CLS]`, `[SEP]`, and more. You can easily add new special tokens to the vocabulary without altering the underlying tokenizer structure. These tokens are used for various purposes such as marking the start or end of sentences, padding sequences, and masking tokens for language modeling tasks.

#### Tokenizing Text

The `encode` method in the Tokenizer class is the key for tokenization. It takes the input text and tokenizes it, allowing for various options like adding special tokens, padding, truncation, and controlling the maximum length. For instance, by setting `add_special_tokens=True`, you ensure that the appropriate special tokens are added to the beginning and end of the token sequence. Truncation and padding can be managed using parameters like `max_length`, `padding`, and `truncation`.

```python
text = "Hello, how are you?"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer.encode(text, add_special_tokens=True)
print(encoded)
```

```shell
[101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
```

#### Decoding Tokens

The `decode` method performs the inverse operation of tokenization. It takes a sequence of token IDs and converts it back to a human-readable text string. Special tokens can be skipped during decoding using the `skip_special_tokens` parameter.

```python
decoded_text = tokenizer.decode(encoded, skip_special_tokens=True)
print(decoded_text)
```

#### Batch Encoding

For efficiency in processing multiple inputs, the `batch_encode_plus` method tokenizes a batch of sequences at once. It returns a `BatchEncoding` object that contains various model inputs like `input_ids`, `attention_mask`, and more.

```python
texts = ["Hello, how are you?", "I'm doing well, thank you!"]
batch_encoding = tokenizer.batch_encode_plus(texts, add_special_tokens=True, padding=True, truncation=True)
print(batch_encoding.input_ids)
print(batch_encoding.attention_mask)
```

#### Custom Tokenization

Hugging Face tokenizers also support custom tokenization functions. If your input is already tokenized (e.g., for Named Entity Recognition), you can set `is_split_into_words=True` and tokenize accordingly.

```python
tokens = ["Hello", ",", "how", "are", "you", "?"]
encoded = tokenizer.encode(tokens, is_split_into_words=True)
print(encoded)
```

Hugging Face's Tokenizer class simplifies this process by providing an array of methods and options to tokenize, encode, decode, and handle special tokens effectively. By understanding and utilizing these strategies, you can seamlessly integrate tokenization into your NLP workflows.

## Section 2: Fine-Tune a Pretrained Model with the Trainer Class

Get ready to dive deeper into training using the powerful `Trainer` class. By exploring hyperparameter tuning, visualization options, and advanced training techniques, you'll wield Hugging Face's prowess to the fullest.

### Hyperparameter Tuning

Mastering hyperparameters is a cornerstone of model training. Uncover essential hyperparameters within the `TrainingArguments` class:

- `learning_rate`: Controls optimization's step size.
- `num_train_epochs`: Specifies training epochs.
- `per_device_train_batch_size`: Sets batch size during training.

Discover the art of hyperparameter tuning by experimenting with different values for optimal performance.

### Visualizing Training Progress

Real-time training visualization empowers your progress tracking. Integrate TensorBoard seamlessly into the `Trainer` class for dynamic insights:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=500,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

Visualizing training progress using TensorBoard enhances your ability to track metrics, identify trends, and make informed decisions.

### Advanced Training Techniques

Elevate your training beyond the basics by exploring advanced techniques like gradient accumulation, mixed precision training, and learning rate schedules:

```python
# Experiment with gradient accumulation and mixed precision training
training_args.gradient_accumulation_steps = 4
training_args.fp16 = True

# Implement a learning rate schedule
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=5e-

5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
```

Understanding and implementing these advanced techniques empower you to achieve faster convergence and better results in complex training scenarios.

## Section 3: Fine-Tuning a Pretrained Model in Native PyTorch

Delve into the heart of model training by exploring native PyTorch techniques. By mastering gradient clipping, learning rate schedulers, and custom evaluation functions, you'll deepen your understanding of underlying mechanisms.

### Gradient Clipping

Prevent gradient explosions with precision. Implement gradient clipping to scale gradients if their norms surpass a predefined threshold:

```python
from torch.nn.utils import clip_grad_norm_

max_grad_norm = 1.0
clip_grad_norm_(model.parameters(), max_grad_norm)
```

Implementing gradient clipping safeguards your training from numerical instability and ensures smoother convergence.

### Learning Rate Schedulers

Fine-tune your model's learning rate dynamically. PyTorch offers a suite of learning rate schedulers, including StepLR, ReduceLROnPlateau, and CosineAnnealingLR:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler.step()
```

Adapting learning rates using these schedulers optimizes model performance and accelerates convergence.

### Custom Evaluation Functions

Elevate your evaluation metrics by incorporating precision, recall, and F1-score alongside accuracy:

```python
from sklearn.metrics import precision_recall_fscore_support

def compute_custom_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": (predictions == labels).mean(), "precision": precision, "recall": recall, "f1": f1}
```

Crafting custom evaluation functions enables you to assess model performance with a nuanced perspective.

## Section 4: Additional Resources

As you've delved deep into Hugging Face's capabilities, keep expanding your horizons with these additional resources:

- [Hugging Face Model Hub](https://huggingface.co/models): Access a treasure trove of pretrained models for diverse NLP tasks.
- [Transformers Documentation](https://huggingface.co/docs/transformers/): Journey into the library's comprehensive documentation for advanced features and use cases.
- [Hugging Face Forum](https://discuss.huggingface.co/): Join a vibrant community to share, learn, and engage with fellow NLP enthusiasts.

## Conclusion

Congratulations! You've completed an extensive journey into the world of advanced NLP using the Hugging Face library. By mastering data preparation, advanced training techniques, and native PyTorch approaches, you're equipped to tackle complex NLP challenges with confidence. Remember, continuous learning and exploration in this dynamic field will lead you to ever greater heights. Happy fine-tuning and experimenting!