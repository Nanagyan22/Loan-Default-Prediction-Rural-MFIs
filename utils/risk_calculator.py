import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging

class RiskCalculator:
    """Calculate various risk metrics and assessments"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.20,
            'medium': 0.50,
            'high': 0.75
        }
    
    def calculate_risk_band(self, probability: float) -> str:
        """Calculate risk band based on default probability"""
        if probability <= self.risk_thresholds['low']:
            return "Low Risk"
        elif probability <= self.risk_thresholds['medium']:
            return "Medium Risk" 
        elif probability <= self.risk_thresholds['high']:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def calculate_confidence_interval(self, probability: float, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for probability estimate"""
        # Simple approximation for confidence interval
        margin = 1.96 * np.sqrt(probability * (1 - probability) / 100)  # Assuming sample size of 100
        lower = max(0, probability - margin)
        upper = min(1, probability + margin)
        return lower, upper
    
    def assess_portfolio_risk(self, probabilities: List[float]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        probabilities = np.array(probabilities)
        
        risk_bands = [self.calculate_risk_band(p) for p in probabilities]
        risk_distribution = pd.Series(risk_bands).value_counts()
        
        assessment = {
            'total_loans': len(probabilities),
            'average_risk': float(np.mean(probabilities)),
            'median_risk': float(np.median(probabilities)),
            'risk_distribution': risk_distribution.to_dict(),
            'high_risk_percentage': len([p for p in probabilities if p > 0.5]) / len(probabilities) * 100,
            'expected_defaults': int(np.sum(probabilities)),
            'risk_score': self._calculate_portfolio_risk_score(probabilities)
        }
        
        return assessment
    
    def _calculate_portfolio_risk_score(self, probabilities: np.ndarray) -> str:
        """Calculate overall portfolio risk score"""
        avg_risk = np.mean(probabilities)
        risk_variance = np.var(probabilities)
        
        # Combine average risk and variance for overall score
        if avg_risk < 0.3 and risk_variance < 0.05:
            return "Conservative"
        elif avg_risk < 0.5 and risk_variance < 0.1:
            return "Moderate"
        elif avg_risk < 0.7:
            return "Aggressive"
        else:
            return "High Risk"
    
    def calculate_loan_pricing_adjustment(self, default_probability: float, base_rate: float = 0.15) -> Dict[str, float]:
        """Calculate risk-adjusted pricing"""
        # Simple risk-based pricing model
        risk_premium = default_probability * 0.5  # 50% of expected loss as premium
        adjusted_rate = base_rate + risk_premium
        
        return {
            'base_rate': base_rate,
            'risk_premium': risk_premium,
            'adjusted_rate': adjusted_rate,
            'rate_increase_percentage': (risk_premium / base_rate) * 100
        }
    
    def generate_risk_alerts(self, probability: float, borrower_data: Dict[str, Any]) -> List[str]:
        """Generate risk alerts based on probability and borrower characteristics"""
        alerts = []
        
        if probability > 0.75:
            alerts.append("🚨 CRITICAL: Very high default risk detected")
        elif probability > 0.5:
            alerts.append("⚠️ WARNING: High default risk")
        
        # Additional rule-based alerts
        if borrower_data.get('previous_default_count', 0) > 0:
            alerts.append("⚠️ Previous default history detected")
        
        if borrower_data.get('loan_amount_requested_ghs', 0) > borrower_data.get('monthly_income_ghs', 0) * 10:
            alerts.append("⚠️ High loan-to-income ratio")
        
        if borrower_data.get('gps_distance_to_town_km', 0) > 50:
            alerts.append("ℹ️ Remote location - consider collection logistics")
        
        return alerts
    
    def calculate_business_impact(self, probability: float, loan_amount: float) -> Dict[str, float]:
        """Calculate potential business impact of the loan"""
        expected_loss = probability * loan_amount
        
        # Simple profit calculation
        base_profit_rate = 0.10  # 10% profit margin
        expected_profit = loan_amount * base_profit_rate * (1 - probability)
        
        net_expected_value = expected_profit - expected_loss
        
        return {
            'loan_amount': loan_amount,
            'expected_loss': expected_loss,
            'expected_profit': expected_profit,
            'net_expected_value': net_expected_value,
            'risk_adjusted_return': net_expected_value / loan_amount if loan_amount > 0 else 0
        }
