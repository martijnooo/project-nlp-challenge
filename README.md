# Natural Language Processing Challenge: Fake vs Real News Classification

## Project Overview
This project aims to build a classifier to distinguish between **real and fake news** using the text of news articles. The dataset includes the following columns:

- `label`: 0 = fake, 1 = real
- `title`: headline of the article
- `text`: full article content
- `subject`: category/topic of the news
- `date`: publication date

The goal is to train a model using text embeddings and additional features, and then predict labels for a validation set.

---

## Dataset
- **Training & Test**: `dataset/data.csv` (split into training and test sets)
- **Validation**: `dataset/validation_data.csv` (predictions will be saved in the same format)

---

## Approach

### 1. Data Inspection
- Checked for missing values and label balance
- Inspected text length distribution
- Very little preprocessing required for embeddings

### 2. Feature Engineering
- **Text Embeddings**: Combined `title` + `text` and converted to dense vectors using [SentenceTransformers](https://www.sbert.net/)
- **Article length**: Numeric feature based on number of tokens
- **Subject**: One-hot encoded categorical feature
- Combined features: `[embeddings + length + subject]`

### 3. Handling Long Articles
- Implemented **chunking / sliding window** for articles exceeding token limits
- Generated embeddings for each chunk, then aggregated (e.g., mean)

### 4. Model Training

#### Baseline Models
- **Logistic Regression**
- **Linear Support Vector Classifier (LinearSVC)**
- Split data into `train/test` sets before evaluation
- Evaluated using **accuracy, precision, recall, and F1-score**

#### Hyperparameter Tuning
- Used **GridSearchCV** for Logistic Regression and LinearSVC
- Important parameters tuned:
  - Logistic Regression: `C`, `penalty`, `solver`
  - LinearSVC: `C`, `loss`, `max_iter`

#### Gradient Boosting 
- Explored `GradientBoostingClassifier`
---

## Evaluation

Example metrics from LinearSVC (optimized with GridSearchCV):

**Best parameters:**  
`{'C': 100, 'loss': 'squared_hinge', 'max_iter': 2000}`

**Best CV score:**  
`0.9609`

**Test Accuracy:**  
`0.9608`

**Confusion Matrix:**  
`[[4790 228]`
`[ 163 4805]]`


**Precision:** 0.9547  
**Recall:** 0.9672  
**F1-score:** 0.9600

### Interpretation
- Model performs very well on both classes
- Balanced precision and recall
- Accuracy ~96% on test data

## Saving and Reusing Embeddings
- Precomputed embeddings were saved to `.npy` files
- This avoids recomputing embeddings for repeated experiments

## Libraries Used

- `numpy`  
- `pandas`  
- `scikit-learn`  
  - `LogisticRegression`  
  - `LinearSVC`  
  - `GradientBoostingClassifier`  
  - `GridSearchCV`  
- `sentence_transformers` (for generating text embeddings)  
