# ❤️ Heart Failure Prediction using Machine Learning

## 📖 Project Overview
This project focuses on predicting **heart failure events** using clinical records. Heart disease is a leading cause of death worldwide, and **early detection** can save lives. Using machine learning algorithms, this project analyzes patient data and builds predictive models to determine the **risk of heart failure**.

The notebook implements multiple classification models, evaluates their performance, and compares their effectiveness using standard metrics.

---

## 🗂️ Dataset
The dataset used is the **Heart Failure Clinical Records Dataset**, containing clinical data collected from patients. Key features include:

- Age 👤
- Anaemia 🩸
- Creatinine Phosphokinase (CPK) 🧪
- Diabetes 🍬
- Ejection Fraction ❤️
- High Blood Pressure 🩺
- Platelets 🧫
- Serum Creatinine 🧪
- Serum Sodium 🧂
- Sex 🚻
- Smoking 🚬
- Time ⏱️ (follow-up period)
- Target variable: `DEATH_EVENT` (1 if the patient died, 0 otherwise) ⚠️

> **Source:** [Kaggle Heart Failure Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

---

## 🛠️ Libraries and Tools Used
- **Data Manipulation & Analysis:** `pandas`, `numpy`  
- **Visualization:** `matplotlib`, `seaborn`  
- **Machine Learning:** `scikit-learn`  
  - Models: `LogisticRegression`, `KNeighborsClassifier`, `SVC`, `GaussianNB`, `DecisionTreeClassifier`  
  - Preprocessing: `StandardScaler`, `ColumnTransformer`  
  - Model Tuning: `GridSearchCV`  
- **Evaluation Metrics:** `accuracy_score`, `confusion_matrix`, `classification_report`, `roc_auc_score` ✅

---

## 📝 Project Workflow
1. **Data Loading & Exploration** 🔍
   - Load the dataset and check for missing values.
   - Perform descriptive statistics and exploratory data analysis.
2. **Data Preprocessing** ⚙️
   - Handle missing values (if any).
   - Scale numerical features.
   - Encode categorical variables if necessary.
3. **Model Training** 🤖
   - Split the data into training and testing sets.
   - Train multiple classifiers:
     - Logistic Regression
     - K-Nearest Neighbors
     - Support Vector Machine
     - Naive Bayes
     - Decision Tree
4. **Model Evaluation** 📊
   - Evaluate each model using **accuracy**, **confusion matrix**, **classification report**, and **ROC-AUC score**.
   - Compare models to select the best-performing one.
5. **Hyperparameter Tuning** ⚡
   - Use `GridSearchCV` to find the optimal parameters for models to improve performance.

---

## 📈 Results
- Outputs key metrics for each model.
- Visualization of **confusion matrices** and **ROC curves** to understand model performance.
- Final model selection is based on the balance between **accuracy** and **interpretability**.

---

## 🚀 How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/heart-failure-prediction.git
2. Install required libraries:
   pip install -r requirements.txt
3. Place the dataset heart_failure_clinical_records_dataset.csv in the project folder.
4. Run the notebook:
  jupyter notebook Heart_Failure_Prediction.ipynb
