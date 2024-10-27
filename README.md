## Consumer Complaint Classification

This project involves building and fine-tuning machine learning models to classify consumer complaints into specific product categories. This project was undertaken using various natural language processing (NLP) and machine learning techniques, ultimately achieving a classification accuracy of 83.49% after hyperparameter tuning.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview

The purpose of this project is to classify consumer complaints into appropriate product categories using text classification models. By automating this task, customer support and management teams can gain insights into common complaints, prioritize issues, and streamline responses. The dataset used in this project is sourced from Consumer Financial Protection Bureau.

# Dataset Utilized

The dataset contains consumer complaint narratives and the respective product categories they relate to. Some examples of product categories include "Credit card," "Mortgage," "Debt collection," and "Student loan." Preprocessing steps were applied to ensure data consistency and balance.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Key Achievements

- # Text Preprocessing and Data Augmentation:
  - Employed NLP techniques such as stopword removal, lemmatization, and synonym replacement to enrich and balance underrepresented classes.
- # Feature Extraction:
  - Used TF-IDF vectorization to represent text as numerical data, considering both unigrams and bigrams, resulting in a comprehensive feature set.
- # Model Building and Evaluation:
  - Multiple models were trained, including Support Vector Machine (SVM), Naive Bayes, Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN).
  - The Support Vector Machine (SVM) model achieved the highest accuracy. 
- # Hyperparameter Tuning:
  - Fine-tuned the SVM model with grid search, achieving 83.49% accuracy.

## Model Performance

# Metrics :
- Accuracy: 83.49
- Macro Average:
  - Precision: 0.83
  - Recall: 0.83
  - F1-score: 0.83
- Individual category performance ranges from 0.76 to 0.94 F1-score.

## Key Components

# Data Preprocessing
- Text lowercasing
- Punctuation Removal
- Number removal
- Stop words removal
- Lemmatization
- Confidential Information removal (x's removal)

# Data Balancing

- Sample size: 20,000 per category
- Undersampling for majority classes
- Data Augmentation using synonym replacement for minority classes

# Feature Engineering

- TF-IDF Vectorization
- Bigram features (ngram_range=(1,2))
- Maximum 10,000 features

# Model Training

- Linear SVM with GridSearchCV
- Hyperparameters tuned:
  - C: [0.1, 1, 10]
  - max_iter: [5000,10000]
- Best Parameters: {'C':1, 'max_iter': 5000}

## References

- Naik, P. K., Prashanth, T., Chandru, S., Jaganath, S., & Balan, S. (Year). *Complaint Classification Using Machine Learning & Deep Learning*.

This project builds on methodologies discussed in the above publication. The insights provided in their work contributed to my development and fine-tuning of the classification models used here.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
