# End-to-End Student Performance Prediction

This project is a **Machine Learning web application using Streamlit** that predicts student final marks based on academic and lifestyle features. It also includes a complete **data science pipeline from data cleaning to model deployment**.

#  Project Overview

The goal of this project is to predict a student's **Final Marks** using features such as:

- Hours Studied  
- Attendance  
- Assignments Score  
- Sleep Hours  
- Previous Marks  
- Internet Usage  
- Extracurricular Activities  
- Mock Test Score  
- Study Efficiency  

# Technologies Used

- Python 
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Streamlit  
- Pickle  


#  Machine Learning Workflow

## 1. Data Collection
Dataset: `student_data_set.csv`

## 2. Data Preprocessing
- Handling missing values using mean
- Removing duplicate records
- Outlier detection using IQR method
- Feature engineering (Study Efficiency)

## 3. Model Building
- Algorithm used: **Linear Regression**
- Feature scaling using StandardScaler
- Train-test split (80/20)

## 4. Model Evaluation
Metrics used:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score


## 5. Model Deployment
- Model saved using `pickle`
- Interactive web app built using Streamlit

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
