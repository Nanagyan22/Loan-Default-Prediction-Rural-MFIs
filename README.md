# Real-Time Loan Default Prediction – Rural MFIs

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/streamlit-1.29-orange)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

🎯 **Project:** Credit Scoring System for Microfinance Institutions (MFIs)  
🏫 **Institution:** University of Ghana  
👤 **Author:** Francis Afful Gyan | ID: 22253332  
📅 **Date:** July 2025  

---

## Table of Contents
- [Project Overview](#project-overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Details](#model-details)  
- [Folder Structure](#folder-structure)  
- [License](#license)  

---

## Project Overview

This project implements a **real-time loan default prediction system** for rural MFIs using machine learning.  
The system uses a **Random Forest Classifier** trained on borrower transactional data to predict the probability of loan default.  

Users can input borrower details via a **Streamlit web interface** and receive:  
- **Probability of default**  
- **Risk classification** (High Risk / Low Risk)  

This helps MFIs make informed lending decisions and manage risk effectively.  

---

## Features

- Real-time loan default prediction  
- Interactive Streamlit interface  
- Input form for borrower features: age, income, loan amount, interest rate, savings, etc.  
- Probability-based risk classification  
- Uses saved preprocessing pipeline to ensure consistency with training data  

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction
