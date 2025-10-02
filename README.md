# 🏦 Customer Churn Prediction

A machine learning web application that predicts customer churn probability using TensorFlow and Streamlit. This tool helps banks and financial institutions identify customers at risk of leaving and take proactive retention measures.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- 
## 🎯 Overview

This application uses a deep learning classification model to predict whether a customer is likely to churn (leave the bank) based on various features including:
- Personal information (age, gender, geography)
- Financial information (credit score, account balance, estimated salary)
- Account details (tenure, number of products, credit card status, active membership)

The model provides a churn probability score and actionable recommendations to help reduce customer attrition.

## ✨ Features

- **Real-time Predictions**: Instant churn probability calculation based on customer inputs
- **Interactive UI**: User-friendly interface built with Streamlit
- **Risk Assessment**: 
  - Churn probability percentage
  - Risk level classification (High/Low)
  - Confidence score
- **Visual Indicators**: 
  - Color-coded results (Red for high risk, Green for low risk)
  - Progress bar visualization
- **Actionable Insights**: Personalized retention strategies based on churn risk
- **Input Summary**: Review all input parameters in an expandable section
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   streamlit==1.28.0
   tensorflow==2.15.0
   pandas==2.1.0
   numpy==1.24.3
   scikit-learn==1.3.0
   pickle-mixin==1.0.2
   ```

4. **Required files**
   Ensure the following files are in your project directory:
   - `model.h5` - Trained TensorFlow model
   - `scaler.pkl` - StandardScaler for feature scaling
   - `label_encoder.pkl` - LabelEncoder for gender encoding
   - `onehot_encoder.pkl` - OneHotEncoder for geography encoding

## 💻 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Make predictions**
   - Fill in customer details in the input fields:
     - **Geography**: Select from France, Germany, or Spain
     - **Gender**: Select Male or Female
     - **Age**: Use slider (18-92 years)
     - **Credit Score**: Enter value (300-850)
     - **Balance**: Enter account balance
     - **Estimated Salary**: Enter annual salary
     - **Tenure**: Select years with bank (0-10)
     - **Number of Products**: Select (1-4)
     - **Has Credit Card**: Yes/No
     - **Is Active Member**: Yes/No
   - Click "Predict Churn" button
   - View churn probability and recommendations

## 📁 Project Structure

```
customer-churn-prediction/
│
├── app.py                          # Main Streamlit application
├── model.h5                        # Trained TensorFlow model
├── scaler.pkl                      # Feature scaler
├── label_encoder.pkl               # Gender encoder
├── onehot_encoder.pkl              # Geography encoder
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── data/                           # Dataset
│   ├── raw/
│   │   └── customer_data.csv
│   └── processed/
│       └── processed_data.csv
│
├── models/                         # Model checkpoints
│   └── checkpoints/
│
└── logs/                           # TensorBoard logs
    └── fit/
```

## 🤖 Model Details

### Architecture
- **Type**: Deep Neural Network (Binary Classification)
- **Framework**: TensorFlow/Keras
- **Input Features**: 12 features (9 numerical + 3 one-hot encoded geography)
- **Output**: Single probability value (0-1) representing churn likelihood
- **Activation**: Sigmoid (output layer)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

### Network Architecture
```
Input Layer (12 features)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Features Used
1. **Numerical Features**:
   - CreditScore (300-850)
   - Age (18-92)
   - Tenure (0-10 years)
   - Balance (account balance)
   - NumOfProducts (1-4)
   - HasCrCard (0/1)
   - IsActiveMember (0/1)
   - EstimatedSalary
   - Gender (encoded: 0/1)

2. **Categorical Features**:
   - Geography (one-hot encoded: France, Germany, Spain)

### Preprocessing Pipeline
1. **Label Encoding**: Gender (Male → 1, Female → 0)
2. **One-Hot Encoding**: Geography (creates 3 binary columns)
3. **Standard Scaling**: Normalize all features to mean=0, std=1

### Performance Metrics
- **Accuracy**: 86.5%
- **Precision**: 84.2%
- **Recall**: 81.7%
- **F1-Score**: 82.9%
- **AUC-ROC**: 0.89

### Training Details
- **Dataset Size**: 10,000 samples
- **Train/Test Split**: 80/20
- **Epochs**: 100
- **Batch Size**: 32
- **Early Stopping**: Patience of 10 epochs
- **Validation Strategy**: Hold-out validation

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow 2.15** - Deep learning framework
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Preprocessing and metrics
- **Pickle** - Model serialization

## 📊 Model Training

To retrain the model with your own data:

1. **Prepare your dataset**
   ```python
   # Required columns:
   # CreditScore, Geography, Gender, Age, Tenure, Balance, 
   # NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited
   ```

2. **Run the training notebook**
   ```bash
   jupyter notebook notebooks/03_model_training.ipynb
   ```

3. **Training script example**:
   ```python
   import tensorflow as tf
   from tensorflow import keras
   
   # Build model
   model = keras.Sequential([
       keras.layers.Dense(64, activation='relu', input_shape=(12,)),
       keras.layers.Dropout(0.2),
       keras.layers.Dense(32, activation='relu'),
       keras.layers.Dropout(0.2),
       keras.layers.Dense(16, activation='relu'),
       keras.layers.Dense(1, activation='sigmoid')
   ])
   
   # Compile
   model.compile(
       optimizer='adam',
       loss='binary_crossentropy',
       metrics=['accuracy']
   )
   
   # Train
   history = model.fit(
       X_train, y_train,
       validation_data=(X_test, y_test),
       epochs=100,
       batch_size=32,
       callbacks=[early_stopping, tensorboard]
   )
   
   # Save
   model.save('model.h5')
   ```

## 📈 Results Interpretation

### Churn Probability
- **0-30%**: Low risk - Customer likely to stay
- **30-50%**: Medium risk - Monitor customer engagement
- **50-70%**: High risk - Immediate action recommended
- **70-100%**: Critical risk - Urgent intervention required

### Recommended Actions

**For High-Risk Customers (>50%)**:
- Reach out with personalized retention offers
- Investigate customer satisfaction issues
- Offer loyalty rewards or incentives
- Schedule account review meetings

**For Low-Risk Customers (<50%)**:
- Continue excellent service
- Consider upselling opportunities
- Maintain regular engagement
- Reward loyalty

## 🔮 Future Enhancements

- [ ] Add SHAP values for model interpretability
- [ ] Implement batch prediction from CSV upload
- [ ] Add customer segmentation analysis
- [ ] Create automated email alerts for high-risk customers
- [ ] Integrate with CRM systems
- [ ] Add A/B testing for retention strategies
- [ ] Implement time-series analysis for trend detection
- [ ] Add multi-language support
- [ ] Create mobile app version
- [ ] Add dashboard for historical predictions


**⭐ If you find this project useful, please consider giving it a star!**
