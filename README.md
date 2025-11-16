# Sentiment Classification in Amazon Fine Food Reviews

## Overview

This project implements and compares various machine learning and deep learning methodologies for **sentiment classification** using consumer reviews from the **Amazon Fine Food Reviews dataset**. The primary goal is to analyze the sentiment expressed in textual product reviews and classify them as either **positive or negative**, thereby providing insights into consumer feedback.

The analysis compares three distinct methodological approaches: lexicon-based methods, traditional supervised machine learning models (after resampling), and advanced large language models (LLMs) leveraging fine-tuning and prompt-based learning.

## Project Details

| Component | Detail |
| :--- | :--- |
| **Objective** | Analyze and classify sentiment (Positive/Negative) in Amazon product reviews. |
| **Dataset Used** | Amazon Fine Food Reviews (1000 records subset used for this project). |
| **Input Data** | Textual content of the review (`Text`). |
| **Target Variable** | Binary classification derived from the review `Score`. |

## Dataset and Preprocessing

### Dataset Characteristics

The raw dataset contains over 500,000 reviews collected over ten years, ending in October 2012. Each review includes information like UserID, ProductID, Helpfulness metrics, Score, Time (UNIX timestamp), Summary, and Text.

### Data Cleaning and Labeling

1.  **Neutral Review Removal:** Reviews with a `Score=3` (neutral) were removed because their sentiment (positive or negative) was unclear.
2.  **Binary Label Assignment:**
    *   **Positive (Label = 1):** Reviews where `Score > 3`.
    *   **Negative (Label = 0):** Reviews where `Score < 3`.
3.  **Text Cleaning:** The textual content was cleaned by removing HTML tags, keeping only letters (removing numbers and punctuation), converting text to lowercase, tokenizing, removing stopwords (such as "and," "the," "is"), and applying **Lemmatization** to convert words to their root forms.

### Handling Class Imbalance

The dataset suffered from a **severe class imbalance**. On the training split (80% of data), 617 records were positive (label = 1) while only 121 records were negative (label = 0).

To address this bias, a combination of resampling methods was utilized for training traditional ML models:

*   **SMOTE (Synthetic Minority Oversampling Technique):** Used to increase the number of synthetic samples for the minority class (negative reviews, label = 0).
*   **Undersampling:** Used subsequently to reduce the number of samples in the majority class.

### Feature Extraction (Vectorization)

Before applying traditional machine learning models, the cleaned text was converted into a numerical matrix using **TfidfVectorizer** (Term Frequency–Inverse Document Frequency). A maximum of 5000 features were utilized, fitted only on the training set.

## Methodologies Implemented

The project compared four main approaches to sentiment classification:

### 1. Lexicon-Based Sentiment Analysis

This method assesses sentiment by aggregating sentiment scores assigned to individual words based on a dictionary.

*   **VADER (Valence Aware Dictionary and sEntiment Reasoner):** A rule-based tool designed for social media contexts, scoring words from −1 (extremely negative) to +1 (extremely positive).
*   **AFINN:** A lexicon where words are assigned scores from −5 to +5, suitable for short, simple texts.

### 2. Traditional Machine Learning Models (with Resampling)

Models were trained on the resampled, vectorized training data.

*   **Models Included:** Logistic Regression, Random Forest, Naive Bayes, and SVM.

### 3. Transformer-Based Language Models (Fine-Tuning)

Two powerful Transformer architectures, pre-trained on vast amounts of data, were fine-tuned for this specific classification task:

*   **BERT (Bidirectional Encoder Representations from Transformers):** A Transformer-based model trained bidirectionally to understand context by masking words (Masked Language Modeling) and predicting subsequent sentences (Next Sentence Prediction).
*   **RoBERTa (Robustly Optimized BERT Pretraining Approach):** An optimized version of BERT that eliminated the Next Sentence Prediction task, trained on a 10x larger dataset, and used dynamic masking, leading to generally higher accuracy in classification tasks.

### 4. Generative AI (Prompt-Based Learning)

*   **Gemini Flash 2.0:** A modern Google large language model (LLM) was employed without explicit re-training (fine-tuning). It uses a **prompt-based learning** approach, relying on dynamic context understanding to classify sentiment efficiently. This model is optimized for low latency and fast processing of large batches of reviews.

## Results

Performance was evaluated using Accuracy and F1-score. The F1-score is particularly relevant for classification, offering a balanced assessment of precision and recall.

| Model Type | Model Name | Accuracy | F1-score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Generative AI** | **Gemini Flash 2.0** | **0.97** | **0.97** | Best overall performance. |
| LLM (Fine-tuned) | RoBERTa | 0.94 | 0.96 | Highest performing fine-tuned model. |
| LLM (Fine-tuned) | BERT | 0.93 | 0.96 | Strong performance due to deep context understanding. |
| Machine Learning | Random Forest | 0.89 | 0.94 | Highest traditional ML performer. |
| Lexicon-Based | VADER | 0.88 | 0.93 | Strong lexicon baseline. |
| Machine Learning | Logistic Regression | 0.88 | 0.93 | |
| Machine Learning | SVM | 0.89 | 0.92 | |
| Lexicon-Based | AFINN | 0.86 | 0.92 | |
| Machine Learning | Naive Bayes | 0.82 | 0.89 | |

### Key Conclusion

The Gemini Flash 2.0 model achieved the highest accuracy (0.97) and F1-score (0.97), demonstrating that modern generative AI, when integrated via API using effective prompt design, can yield competitive or superior results compared to fine-tuned or traditional ML models. The model's strength lies in its speed, deep semantic understanding, and reliability without the need for intensive re-training.
