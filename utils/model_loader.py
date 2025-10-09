import joblib
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.feature_engineering import FeatureEngineer

class ModelLoader:
    """Handle loading and prediction with trained ML models"""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.models = {}
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all available model artifacts"""
        try:
            # Load preprocessor if available
            preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logging.info("Preprocessor loaded successfully")
            
            # Load only Random Forest + SMOTE pipeline
            model_files = {
                'rf_pipeline': 'rf_pipeline.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.artifacts_dir, filename)
                if os.path.exists(model_path):
                    try:
                        model = joblib.load(model_path)
                        self.models[model_name] = model
                        logging.info(f"Model {model_name} loaded successfully")
                    except Exception as e:
                        logging.error(f"Error loading {model_name}: {str(e)}")
            
            if not self.models:
                logging.warning("No model artifacts found")
                
        except Exception as e:
            logging.error(f"Error loading artifacts: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of Random Forest model"""
        return {
            'Random Forest + SMOTE': 'rf_pipeline' in self.models
        }
    
    def _prepare_input_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert input dictionary to DataFrame with proper feature engineering"""
        return self.feature_engineer.prepare_input(input_data)
    
    def predict_single(self, input_data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Make prediction for single borrower"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        try:
            # Prepare input data
            df = self._prepare_input_data(input_data)
            
            # Get model
            model = self.models[model_name]
            
            # Handle pipelines with SMOTE (transform data manually and use classifier)
            if hasattr(model, 'named_steps') and 'smote' in model.named_steps:
                # Transform data through preprocessor
                preprocessor = model.named_steps['preprocessor']
                X_transformed = preprocessor.transform(df)
                
                # Get classifier (skip SMOTE)
                classifier = model.named_steps['clf']
                probabilities = classifier.predict_proba(X_transformed)
                
                if probabilities.shape[1] > 1:
                    default_prob = probabilities[0][1]  # Probability of default (class 1)
                else:
                    default_prob = probabilities[0][0]
            elif hasattr(model, 'predict_proba'):
                # Regular pipeline with predict_proba
                probabilities = model.predict_proba(df)
                if probabilities.shape[1] > 1:
                    default_prob = probabilities[0][1]  # Probability of default (class 1)
                else:
                    default_prob = probabilities[0][0]
            else:
                # Fallback to predict
                prediction = model.predict(df)
                default_prob = float(prediction[0]) if hasattr(prediction[0], 'item') else prediction[0]
            
            # Calculate risk band
            risk_band = self._calculate_risk_band(default_prob)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(model, df, model_name)
            
            result = {
                'default_probability': float(default_prob),
                'risk_band': risk_band,
                'confidence': 0.85,  # Default confidence
                'model_used': model_name
            }
            
            if feature_importance:
                result['feature_importance'] = feature_importance
            
            return result
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Make predictions for batch of borrowers"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        try:
            # Prepare batch data
            batch_results = []
            
            for idx, row in df.iterrows():
                input_data = row.to_dict()
                try:
                    result = self.predict_single(input_data, model_name)
                    batch_results.append({
                        'applicant_id': input_data.get('applicant_id', f'CUST-{idx:06d}'),
                        'default_probability': result['default_probability'],
                        'risk_band': result['risk_band'],
                        'recommendation': self._get_recommendation(result['default_probability'])
                    })
                except Exception as e:
                    # Handle individual prediction errors
                    batch_results.append({
                        'applicant_id': input_data.get('applicant_id', f'CUST-{idx:06d}'),
                        'default_probability': 0.5,  # Default risk
                        'risk_band': 'Medium Risk',
                        'recommendation': 'REVIEW',
                        'error': str(e)
                    })
            
            results_df = pd.DataFrame(batch_results)
            
            # Merge with original data
            if 'applicant_id' in df.columns:
                results_df = df.merge(results_df, on='applicant_id', how='left')
            else:
                # Add index-based merge
                df_with_id = df.copy()
                df_with_id['applicant_id'] = [f'CUST-{i:06d}' for i in range(len(df))]
                results_df = df_with_id.merge(results_df, on='applicant_id', how='left')
            
            return results_df
            
        except Exception as e:
            logging.error(f"Batch prediction error: {str(e)}")
            raise Exception(f"Batch prediction failed: {str(e)}")
    
    def _calculate_risk_band(self, probability: float) -> str:
        """Calculate risk band based on default probability"""
        if probability <= 0.20:
            return "Low Risk"
        elif probability <= 0.50:
            return "Medium Risk" 
        elif probability <= 0.75:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _clean_feature_name(self, name: str) -> str:
        """Clean feature name by removing prefixes and formatting"""
        # Remove num__, cat__, remainder__ prefixes
        for prefix in ['num__', 'cat__', 'remainder__']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Replace underscores with spaces and title case
        name = name.replace('_', ' ').title()
        return name
    
    def _get_feature_importance(self, model, df: pd.DataFrame, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """Extract feature importance from model"""
        try:
            # For pipelines, get feature names from preprocessor
            if hasattr(model, 'named_steps'):
                # Get preprocessor to extract feature names
                preprocessor = model.named_steps.get('preprocessor')
                if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        raw_feature_names = preprocessor.get_feature_names_out().tolist()
                        # Clean feature names
                        feature_names = [self._clean_feature_name(name) for name in raw_feature_names]
                    except:
                        feature_names = [f'feature_{i}' for i in range(100)]  # Fallback
                else:
                    feature_names = [f'feature_{i}' for i in range(100)]
                
                # Get the classifier
                classifier = None
                for step_name in ['clf', 'classifier', 'smote', 'model']:
                    if step_name in model.named_steps:
                        # If it's SMOTE, skip to next step
                        if step_name == 'smote':
                            continue
                        classifier = model.named_steps[step_name]
                        break
                
                # Try to get last step if still no classifier
                if classifier is None:
                    classifier = list(model.named_steps.values())[-1]
                
                # Extract importance from classifier
                if hasattr(classifier, 'feature_importances_'):
                    # Random Forest
                    importances = classifier.feature_importances_
                    importance_data = [
                        {'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                         'importance': float(importances[i])}
                        for i in range(len(importances))
                    ]
                    importance_data.sort(key=lambda x: abs(x['importance']), reverse=True)
                    return importance_data[:8]  # Top 8 features
                
                elif hasattr(classifier, 'coef_'):
                    # Logistic Regression
                    coefficients = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
                    importance_data = [
                        {'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                         'importance': float(coefficients[i])}
                        for i in range(min(len(coefficients), len(feature_names)))]
                    
                    importance_data.sort(key=lambda x: abs(x['importance']), reverse=True)
                    return importance_data[:8]  # Top 8 features
            
            return None
            
        except Exception as e:
            logging.warning(f"Could not extract feature importance: {str(e)}")
            return None
    
    def _get_recommendation(self, default_prob: float) -> str:
        """Get recommendation based on default probability"""
        if default_prob < 0.20:
            return "APPROVE"
        elif default_prob < 0.50:
            return "REVIEW"
        elif default_prob < 0.75:
            return "MANUAL REVIEW"
        else:
            return "REJECT"
