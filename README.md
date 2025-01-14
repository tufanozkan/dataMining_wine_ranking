# Spanish Wine Quality Analysis Project

## Project Overview

This project focuses on analyzing a dataset of Spanish wines, leveraging machine learning techniques to classify wine quality into distinct categories. The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical features. The project includes various machine learning algorithms and evaluates their performance to identify the most effective model for wine quality prediction.

## Objectives

- Preprocess the dataset to ensure data quality and consistency.
- Implement and evaluate multiple machine learning models, including:
  - Decision Trees
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - Gradient Boosting Machines (GBM)
  - AdaBoost
- Perform hyperparameter tuning and depth analysis to optimize model performance.
- Visualize results with learning curves, confusion matrices, and model-specific insights.

---

## Features and Technologies

### Dataset
The dataset, obtained from [Kaggle](https://www.kaggle.com/code/fedesoriano/spanish-wine-quality-dataset-introduction/notebook?select=wines_SPA.csv), contains key attributes of Spanish wines, such as:
- Winery, wine name, and region
- Acidity, body, and price
- User reviews and ratings

### Data Preprocessing
- **Handling Missing Values**:
  - Numerical columns are imputed with median values.
  - Categorical columns are encoded using `LabelEncoder`.
- **Feature Scaling**:
  - Used `StandardScaler` for numerical feature standardization.
- **Target Variable**:
  - Created a new target variable (`rating_category`) to classify wine quality into four categories: Good, Very Good, Excellent, and Superior.

### Machine Learning Models
#### 1. **Decision Trees**
   - Depth analysis to balance bias and variance.
   - Visualized tree structures using Graphviz for interpretability.

#### 2. **K-Nearest Neighbors (KNN)**
   - Evaluated various `k` values to find optimal performance.
   - Used cross-validation for robust results.

#### 3. **Support Vector Machines (SVM)**
   - Compared linear and RBF kernels.
   - Tuned hyperparameters for kernel-specific performance optimization.

#### 4. **Gradient Boosting Machines (GBM)**
   - Demonstrated impact of learning rate and depth on performance.

#### 5. **AdaBoost**
   - Explored weak learners and the effect of boosting iterations.

---

## Performance Evaluation

### Metrics
- Accuracy, Precision, Recall, and F1 Score
- Confusion Matrices for detailed error analysis
- Learning Curves to assess overfitting and underfitting

### Key Insights
- Identified the best-performing model using cross-validation and test set accuracy.
- Visualized feature importance for interpretability (where applicable).

---

## Project Workflow

1. **Data Exploration and Preprocessing**  
   - Cleaned the dataset by addressing missing values and encoding categorical features.
   - Scaled numerical features to standardize inputs for machine learning models.

2. **Model Training and Evaluation**  
   - Trained each algorithm with optimized hyperparameters.
   - Conducted detailed performance evaluations with multiple metrics.

3. **Visualization**  
   - Generated plots for learning curves, confusion matrices, and model-specific analyses.
   - Saved visualizations in the `output_png` directory.

---

## Tools and Technologies

- **Programming Language**: Python  
- **Libraries**:  
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`  
  - Machine Learning: `scikit-learn`  
  - Export & Visualization: `pydotplus`, `Graphviz`  

- **Key Techniques**:  
  - Cross-validation for performance robustness  
  - Hyperparameter tuning to optimize models  
  - Feature scaling and encoding for improved data quality  

---

## How to Run the Project

1. Clone the repository and install required dependencies:  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   pip install -r requirements.txt

2. Download the dataset and place it in the data folder.
Ensure the file is named wines_SPA.csv.

3. Execute the main script:
   ```bash
   python main.py

4. Check the output_png folder for generated plots and visualizations.

---

## Future Enhancements
  - Integrate advanced machine learning algorithms like Random Forests or XGBoost.
  - Perform feature engineering for additional insights.
  - Explore deep learning models for multi-class classification.

