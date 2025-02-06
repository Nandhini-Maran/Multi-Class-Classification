# 📌 Multi-Class Classification Analysis for Diabetes Prediction 🔍

## 📖 Overview
This project performs a **multi-class classification analysis** on a diabetes dataset. The workflow includes:  
✔️ Data preprocessing  
✔️ Exploratory data analysis (EDA)  
✔️ Feature engineering  
✔️ Model building using multiple classifiers  

The primary objective is to predict **diabetes progression** using machine learning techniques. 🚀

## 📂 Dataset
- 📄 **Dataset:** `Diabetes.csv`
- 🔢 **Attributes:** Health & lifestyle factors
- 🎯 **Target Variable:** `Diabetes_012`
  - `0️⃣`: No diabetes
  - `1️⃣`: Pre-diabetes
  - `2️⃣`: Diabetes

## 🔄 Steps in the Analysis
### 1️⃣ Data Preprocessing
🧹 **Cleaning:** Removing duplicates & handling missing values  
🗂️ **Feature Selection:** Dropped `Education` & `Income`  
📊 **Outlier Removal:** IQR method  
📏 **Normalization:** MinMaxScaler  
⚖️ **Class Balancing:** SMOTE (Synthetic Minority Over-sampling Technique)

### 2️⃣ Exploratory Data Analysis (EDA)
📈 **Visualization Techniques:**
- Histograms 📊
- Box plots 📦
- Correlation heatmaps 🔥
- Class distribution analysis 📌

### 3️⃣ Model Building 🏗️
Implemented & evaluated:
- 🌲 **Random Forest Classifier**
- 🎯 **XGBoost Classifier**
- 🚀 **Gradient Boosting Classifier**
- 💡 **LightGBM Classifier**

### 4️⃣ Model Evaluation 🏆
Metrics Used:

✅ **Accuracy**  
✅ **Classification Report**  
✅ **Confusion Matrix**  
✅ **ROC-AUC Score**  

### 5️⃣ Hyperparameter Tuning ⚙️
🔧 **Optimized Models Using `RandomizedSearchCV`** for:
- Random Forest 🌲
- XGBoost 🎯
- Gradient Boosting 🚀
- LightGBM 💡

## 📊 Results
⭐ **Best Performing Model:** LightGBM (83.7% Accuracy)  
🎯 **After Hyperparameter Tuning:** All models achieved ~84% accuracy  
⚠️ **Further tuning may cause overfitting!**

## 💻 Installation & Requirements
📦 Required Libraries:  
Save dependencies in `requirements.txt` and install using:
```bash
pip install -r requirements.txt
```
## 🚀 Usage
1️⃣ Load the dataset (`Diabetes.csv`) 📂  
2️⃣ Run the preprocessing steps ⚙️  
3️⃣ Train different ML models 🤖  
4️⃣ Evaluate model performance 📈  
5️⃣ Use the best model for prediction 🔍  

## 🏁 Conclusion
✔️ **LightGBM performed the best** ✅  
✔️ **High accuracy achieved without overfitting** 🎯  
✔️ **A great approach to diabetes prediction using ML** 🏆  
