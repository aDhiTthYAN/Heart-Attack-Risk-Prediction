# Heart Attack Risk Prediction

## Overview

This project aims to predict the risk of heart attack based on various medical and lifestyle factors using machine learning techniques. The prediction model can help in early diagnosis and preventive measures for individuals at risk of heart attack.

## Dataset

The dataset includes the following features:
- Age
- Cholesterol
- Heart Rate
- Diabetes
- Family History
- Smoking
- Obesity
- Alcohol Consumption
- Exercise Hours Per Week
- Previous Heart Problems
- Medication Use
- Stress Level
- Sedentary Hours Per Day
- BMI
- Triglycerides
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Sex (Male)
- Diet (Healthy and Unhealthy)

## Methodology

The following steps were taken to build the heart attack risk prediction model:

1. **Data Preprocessing**:
   - Handling missing values.
   - Feature scaling using `StandardScaler`.
   - Encoding categorical variables.

2. **Feature Engineering**:
   - Creating new features from existing data.
   - Selecting the most important features using `SelectFromModel`.

3. **Model Building**:
   - Implementing several classification algorithms including RandomForest, SVM, Logistic Regression, XGBoost, LightGBM, and ensemble Voting Classifier.
   - Hyperparameter tuning to optimize model performance.

4. **Model Evaluation**:
   - Evaluating models using metrics like accuracy, precision, recall, and F1-score.
   - Using confusion matrix to assess model performance.

5. **Handling Imbalanced Data**:
   - Applying SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

## Model Performance

## Model Performance

The RandomForest model was optimized with the following parameters:

- `n_estimators=200`
- `max_depth=10`  # reduced depth
- `min_samples_split=10`  # increased minimum samples to split
- `min_samples_leaf=5`  # increased minimum samples at leaf
- `max_features='sqrt'`  # use sqrt of the number of features
- `class_weight='balanced'`  # account for any imbalance
- `random_state=42`

### Evaluation Metrics

The performance of the model on the training and test sets is as follows:

#### Training Set:
- **Accuracy**: 0.8704
- **Precision**: 0.8570
- **Recall**: 0.8888
- **F1 Score**: 0.8726

#### Test Set:
- **Accuracy**: 0.6406
- **Precision**: 0.6427
- **Recall**: 0.6393
- **F1 Score**: 0.6410

#### Test Confusion Matrix:
[[1079 602]
[ 611 1083]]

## How to Use

To use the model to predict heart attack risk for new data, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/heart-attack-risk-prediction.git
    cd heart-attack-risk-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Load the trained model and scaler:
    ```python
    import pickle
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    with open('rf1_model.pkl', 'rb') as file:
        rf1 = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    ```

4. Create a DataFrame with the new data and make predictions:
    ```python
    new_data = pd.DataFrame({
        'Age': [35],
        'Cholesterol': [180],
        'Heart Rate': [70],
        'Diabetes': [0],
        'Family History': [0],
        'Smoking': [0],
        'Obesity': [0],
        'Alcohol Consumption': [0],
        'Exercise Hours Per Week': [5],
        'Previous Heart Problems': [0],
        'Medication Use': [0],
        'Stress Level': [3],
        'Sedentary Hours Per Day': [2],
        'BMI': [22.5],
        'Triglycerides': [130],
        'systolic_bp': [120],
        'diastolic_bp': [80],
        'Sex_Male': [1],
        'Diet_Healthy': [1],
        'Diet_Unhealthy': [0]
    })

    numerical_features = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 'Stress Level',
                          'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 'systolic_bp', 'diastolic_bp']
    
    new_data[numerical_features] = scaler.transform(new_data[numerical_features])
    
    predicted_class = rf1.predict(new_data)
    print("Predicted class:", predicted_class[0])
    ```

## Conclusion

This project demonstrates the use of machine learning to predict the risk of heart attack based on a variety of factors. The RandomForest model was found to be the most effective in this context. Future improvements could include the addition of more data, further feature engineering, and the exploration of other advanced machine learning techniques.


