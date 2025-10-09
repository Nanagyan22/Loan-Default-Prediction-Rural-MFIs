# Real-Time Loan Default Prediction System for Rural MFIs

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/streamlit-1.29-orange)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

🎯 **Project:** Credit Scoring System for Microfinance Institutions (MFIs)  
🏫 **Institution:** University of Ghana  
👤 **Author:** Francis Afful Gyan | ID: 22253332  
📅 **Date:** October 2025  

A machine learning-powered web application designed to assist Microfinance Institutions (MFIs) in rural areas with real-time credit risk assessment and loan default prediction.

## 🎯 Project Overview

This project implements an intelligent credit scoring system specifically tailored for rural Microfinance Institutions. The system leverages machine learning algorithms to predict the likelihood of loan defaults, enabling MFIs to make data-driven lending decisions while managing financial risk effectively.

The application uses a **Random Forest Classifier** trained on comprehensive borrower transactional data to assess creditworthiness and predict default probability. Through an intuitive Streamlit web interface, loan officers can input borrower information and receive instant risk assessments, streamlining the loan approval process while maintaining robust risk management standards.

## ✨ Key Features

**Real-Time Predictions**
- Instant loan default probability calculation
- Risk classification (High Risk/Low Risk) based on configurable thresholds
- Processing time under 2 seconds per prediction

**User-Friendly Interface**
- Clean, intuitive Streamlit-based web application
- Form-based input system for borrower details
- Clear visualization of prediction results and risk metrics
- Mobile-responsive design for field officers

**Comprehensive Input Parameters**
- Demographic data: Age, employment status, education level
- Financial metrics: Income, existing savings, loan amount requested
- Loan terms: Interest rate, loan tenure, payment frequency
- Credit history: Previous loan performance, delinquency records

**Risk Assessment Output**
- Default probability score (0-100%)
- Binary risk classification with confidence levels
- Key risk factors contributing to the prediction
- Recommended actions for different risk levels

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.28.0
- **Backend**: Python 3.9+
- **Machine Learning**: scikit-learn 1.3.0
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/loan-default-prediction-rural.git
cd loan-default-prediction-rural
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download the model files**:
   - Ensure `rf_model.pkl` (Random Forest model) is in the project root
   - Ensure `scaler.pkl` (preprocessing scaler) is in the project root

## 🚀 Usage

### Running Locally

1. **Start the Streamlit application**:
```bash
streamlit run app.py
```

2. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:8501`

### Using the Application

1. **Input Borrower Information**: Fill in all required fields in the form:
   - Personal details (age, employment status)
   - Financial information (income, savings)
   - Loan details (amount, interest rate, tenure)

2. **Submit for Prediction**: Click the "Predict Default Risk" button

3. **Review Results**: The system will display:
   - Default probability percentage
   - Risk classification (High/Low)
   - Confidence score
   - Recommended actions

## 📊 Model Details

### Algorithm
The system employs a **Random Forest Classifier**, chosen for its:
- High accuracy in credit risk assessment
- Ability to handle non-linear relationships
- Robustness to outliers
- Feature importance insights

### Performance Metrics
- **Accuracy**: 86%
- **Recall**: 74% (minimizing false negatives - missed defaulters)
- **Precision**: 62%
- **F1-Score**: 0.68
- **AUC-ROC**: 0.82

### Training Data
The model was trained on historical loan data including:
- 10,000+ loan records from rural MFIs
- 12 input features covering demographics, financials, and loan terms
- Balanced dataset with appropriate handling of class imbalance

### Feature Importance
Top predictive features identified:
1. Debt-to-income ratio
2. Previous loan delinquency count
3. Loan amount to income ratio
4. Age of oldest credit line
5. Employment stability (years at current job)

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```
MODEL_PATH=rf_model.pkl
SCALER_PATH=scaler.pkl
CONFIDENCE_THRESHOLD=0.5
```

### Model Parameters
Adjustable parameters in `config.py`:
- Risk threshold (default: 0.5)
- Feature scaling method
- Model update frequency

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Author

**Francis Afful Gyan**  
Student ID: 22253332  
University of Ghana  
📧 Email: francis.gyan@ug.edu.gh  
🔗 LinkedIn: [https://www.linkedin.com/in/francis-afful-gyan-2b27a5153/]  
📅 Date: October 2025

## 🙏 Acknowledgments

- University of Ghana, Department of Computer Science
- Rural MFI partners for providing domain expertise
- The Streamlit team for the excellent web framework
- scikit-learn community for machine learning tools
- Open-source contributors to the credit scoring domain

## 📚 References

- Basel Committee on Banking Supervision guidelines on credit risk management
- Microfinance best practices for rural lending
- Machine learning applications in financial inclusion

---

**🌐 Live Demo**: [https://loan-default-prediction-rural.streamlit.app/](https://loan-default-prediction-rural.streamlit.app/)

**📊 Project Status**: Active Development

**⭐ If you find this project useful, please consider giving it a star!**
