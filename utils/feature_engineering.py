import pandas as pd
import numpy as np
from typing import Dict, Any

class FeatureEngineer:
    """Handle feature engineering consistent with training notebook"""
    
    # Feature definitions from training notebook
    NUMERIC_FEATURES = [
        'Age', 'Monthly_Income', 'Savings_Balance', 'Loan_Amount',
        'Loan_to_Income_Ratio', 'Savings_to_Income_Ratio', 'Debt_to_Income_Ratio',
        'Credit_Score_Internal', 'Late_Payments_PastYear', 'Other_Debts', 'Missed_Meetings'
    ]
    
    CATEGORICAL_FEATURES = [
        'Gender', 'Marital_Status', 'Education_Level', 'Region', 'Occupation',
        'Employment_Status', 'House_Ownership', 'Bank_Account',
        'Financial_Literacy_Level', 'Community_Reputation'
    ]
    
    ENGINEERED_FEATURES = [
        'High_LTI', 'Low_Savings_Flag', 'High_Debt_Stress',
        'Stable_Employment', 'Dependents_per_1000GHS', 'LTI_Missed_interaction'
    ]
    
    @staticmethod
    def prepare_input(data: Dict[str, Any]) -> pd.DataFrame:
        """Convert input dictionary to DataFrame with proper feature engineering"""
        df = pd.DataFrame([data])
        
        # Map user-friendly field names to model feature names
        field_mapping = {
            'age': 'Age',
            'gender': 'Gender',
            'marital_status': 'Marital_Status',
            'education_level': 'Education_Level',
            'occupation': 'Occupation',
            'region': 'Region',
            'monthly_income_ghs': 'Monthly_Income',
            'loan_amount_requested_ghs': 'Loan_Amount',
            'loan_purpose': 'Loan_Purpose',
            'loan_term_months': 'Loan_Term_Months',
            'previous_default_count': 'Previous_Default_Count',
            'household_size': 'Dependents',
            'savings_balance': 'Savings_Balance',
            'late_payments_pastyear': 'Late_Payments_PastYear',
            'loan_to_income_ratio': 'Loan_to_Income_Ratio',
            'debt_to_income_ratio': 'Debt_to_Income_Ratio'
        }
        
        # Rename columns
        for old_name, new_name in field_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Set default values for missing features
        defaults = {
            'Age': 35,
            'Gender': 'Male',
            'Marital_Status': 'Married',
            'Education_Level': 'JHS',
            'Region': 'Ashanti',
            'Occupation': 'Farmer',
            'Monthly_Income': 1500,
            'Savings_Balance': 500,
            'Loan_Amount': 5000,
            'Loan_to_Income_Ratio': 3.33,
            'Savings_to_Income_Ratio': 0.33,
            'Debt_to_Income_Ratio': 1.0,
            'Credit_Score_Internal': 600,
            'Late_Payments_PastYear': 0,
            'Other_Debts': 0,
            'Missed_Meetings': 0,
            'Employment_Status': 'Self-Employed',
            'House_Ownership': 'Rented',
            'Bank_Account': 'No',
            'Financial_Literacy_Level': 'Medium',
            'Community_Reputation': 'Good',
            'Dependents': 3
        }
        
        # Fill missing values
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)
        
        # Calculate ratio features if not present
        if 'Loan_to_Income_Ratio' not in df.columns or pd.isna(df['Loan_to_Income_Ratio'].iloc[0]):
            df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / (df['Monthly_Income'] + 1e-6)
        
        if 'Savings_to_Income_Ratio' not in df.columns or pd.isna(df['Savings_to_Income_Ratio'].iloc[0]):
            df['Savings_to_Income_Ratio'] = df['Savings_Balance'] / (df['Monthly_Income'] + 1e-6)
        
        if 'Debt_to_Income_Ratio' not in df.columns or pd.isna(df['Debt_to_Income_Ratio'].iloc[0]):
            df['Debt_to_Income_Ratio'] = df['Other_Debts'] / (df['Monthly_Income'] + 1e-6)
        
        # Engineer features (matching training notebook exactly)
        df['High_LTI'] = (df['Loan_to_Income_Ratio'] > 5).astype(int)
        df['Low_Savings_Flag'] = ((df['Savings_to_Income_Ratio'] < 0.1) | (df['Savings_Balance'].isna())).astype(int)
        df['High_Debt_Stress'] = (df['Debt_to_Income_Ratio'] > 3).astype(int)
        df['Stable_Employment'] = df['Employment_Status'].apply(lambda x: 1 if x == 'Salaried' else 0)
        df['Dependents_per_1000GHS'] = df['Dependents'] / (df['Monthly_Income']/1000 + 1e-6)
        df['LTI_Missed_interaction'] = df['Loan_to_Income_Ratio'].fillna(0) * df['Missed_Meetings']
        
        # Select features in correct order
        all_features = (
            FeatureEngineer.NUMERIC_FEATURES + 
            FeatureEngineer.CATEGORICAL_FEATURES + 
            FeatureEngineer.ENGINEERED_FEATURES
        )
        
        # Ensure all features exist
        for feature in all_features:
            if feature not in df.columns:
                if feature in defaults:
                    df[feature] = defaults[feature]
                elif feature in FeatureEngineer.ENGINEERED_FEATURES:
                    df[feature] = 0  # Default engineered features to 0
                else:
                    df[feature] = 0  # Safe default
        
        return df[all_features]
    
    @staticmethod
    def prepare_batch(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare batch of inputs with feature engineering"""
        results = []
        for idx, row in df.iterrows():
            try:
                input_dict = row.to_dict()
                engineered_df = FeatureEngineer.prepare_input(input_dict)
                results.append(engineered_df)
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                # Create a default row
                default_dict = {'age': 35, 'gender': 'Male'}
                results.append(FeatureEngineer.prepare_input(default_dict))
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def get_feature_names():
        """Get all feature names in order"""
        return (
            FeatureEngineer.NUMERIC_FEATURES + 
            FeatureEngineer.CATEGORICAL_FEATURES + 
            FeatureEngineer.ENGINEERED_FEATURES
        )
