# 🤖 Best Model Selector – AutoML Web App

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-success?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

An interactive **Streamlit web app** that automatically:
- 🗂️ Preprocesses your dataset  
- 🔎 Trains multiple **Classification & Regression models**  
- ⚡ Performs **hyperparameter tuning** with `RandomizedSearchCV`  
- 📊 Displays **metrics, confusion matrix heatmaps, and best model selection**  

It’s basically your **AutoML buddy** for quick experiments!

---

## ✨ Features
- 📁 **Upload any CSV file** with a sidebar image branding  
- 🔥 **Drop unnecessary columns** to clean the dataset  
- 🎯 Choose between **Classification or Regression** workflows  
- 🏗️ Uses **Pipelines & ColumnTransformer** to prevent data leakage  
- 🧠 Models included:
  - Logistic Regression
  - Decision Tree (Classifier & Regressor)
  - K-Nearest Neighbors (Classifier & Regressor)
  - SVM (SVC & SVR)
  - Random Forest (Classifier & Regressor)
- 🔬 **RandomizedSearchCV** for hyperparameter optimization  
- 📈 Tabs showing:
  - Classification report as a **styled table**
  - **Metric cards** (Accuracy, Precision, Recall, F1 / R², MAE, MSE)
  - **Confusion Matrix heatmap** for classification
- 🏆 Automatically crowns the **Best Model with highest score**  
- 🪄 User-friendly UX: progress bar, toast notifications, images, success prompts  

---

## 🛠️ Tools & Libraries Used

| Category                | Libraries/Tools                          |
|-------------------------|------------------------------------------|
| **Frontend UI**         | [Streamlit](https://streamlit.io/)       |
| **Data Handling**       | [Pandas](https://pandas.pydata.org/)     |
| **Machine Learning**    | [scikit-learn](https://scikit-learn.org/stable/) |
| **Preprocessing**       | `ColumnTransformer`, `Pipeline`, `LabelEncoder`, `OneHotEncoder`, `SimpleImputer`, `StandardScaler` |
| **Hyperparameter Tuning** | `RandomizedSearchCV`                   |
| **Visualization**       | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Metrics & Reports**   | `classification_report`, `accuracy_score`, `mean_squared_error`, `r2_score`, `confusion_matrix` |
| **Other**               | Python `time` for progress bar delays    |

---

## 📸 Screenshots
> Add screenshots or GIFs of:
1. **Upload File Page**  
2. **Model Tabs with Metrics & Confusion Matrix**  
3. **Best Model Result Screen**  

---





