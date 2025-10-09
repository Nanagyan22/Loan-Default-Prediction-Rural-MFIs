import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging

class ModelExplainer:
    """Provide model explanations without SHAP (custom implementation)"""
    
    def __init__(self):
        self.feature_descriptions = {
            'Age': 'Age of the borrower',
            'Monthly_Income': 'Monthly income in Ghana Cedis',
            'Savings_Balance': 'Current savings balance',
            'Loan_Amount': 'Requested loan amount',
            'Loan_to_Income_Ratio': 'Ratio of loan to monthly income',
            'Savings_to_Income_Ratio': 'Ratio of savings to monthly income',
            'Debt_to_Income_Ratio': 'Ratio of existing debt to income',
            'Credit_Score_Internal': 'Internal credit score',
            'Late_Payments_PastYear': 'Number of late payments in past year',
            'Other_Debts': 'Other outstanding debts',
            'Missed_Meetings': 'Number of missed group meetings',
            'Gender': 'Borrower gender',
            'Marital_Status': 'Marital status',
            'Education_Level': 'Highest education level',
            'Region': 'Geographic region in Ghana',
            'Occupation': 'Primary occupation',
            'Employment_Status': 'Employment type',
            'House_Ownership': 'Home ownership status',
            'Bank_Account': 'Has bank account',
            'Financial_Literacy_Level': 'Financial literacy assessment',
            'Community_Reputation': 'Community standing',
            'High_LTI': 'High loan-to-income flag',
            'Low_Savings_Flag': 'Low savings indicator',
            'High_Debt_Stress': 'High debt stress indicator',
            'Stable_Employment': 'Employment stability flag',
            'Dependents_per_1000GHS': 'Dependents per 1000 GHS income',
            'LTI_Missed_interaction': 'Interaction: LTI × Missed meetings'
        }
    
    def create_feature_contribution_plot(self, feature_importance: List[Dict[str, Any]], 
                                        base_prob: float = 0.23) -> go.Figure:
        """Create waterfall-style feature contribution plot"""
        if not feature_importance:
            return None
        
        # Sort by absolute importance
        sorted_features = sorted(feature_importance, key=lambda x: abs(x['importance']), reverse=True)[:10]
        
        # Create waterfall chart showing contribution to default probability
        features = []
        contributions = []
        colors = []
        
        # Start with base rate
        cumulative = base_prob
        
        for feat in sorted_features:
            feat_name = feat['feature'].replace('num__', '').replace('cat__', '')
            # Clean feature name
            for key in self.feature_descriptions:
                if key.lower() in feat_name.lower():
                    feat_name = key
                    break
            
            features.append(feat_name)
            importance = feat['importance']
            
            # Normalize importance to probability contribution (simplified)
            contribution = importance * 0.05  # Scale factor
            contributions.append(contribution)
            
            # Color based on direction
            if contribution > 0:
                colors.append('#dc3545')  # Red for increasing risk
            else:
                colors.append('#28a745')  # Green for decreasing risk
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=features,
            y=contributions,
            marker_color=colors,
            text=[f"{c:+.1%}" for c in contributions],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Contribution: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Contributions to Default Risk",
            xaxis_title="Features",
            yaxis_title="Risk Contribution",
            yaxis_tickformat='.0%',
            height=500,
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    def create_force_plot_alternative(self, feature_importance: List[Dict[str, Any]],
                                     prediction_prob: float, 
                                     base_rate: float = 0.23) -> go.Figure:
        """Create force plot alternative showing push/pull of features"""
        if not feature_importance:
            return None
        
        # Get top features
        top_features = sorted(feature_importance, key=lambda x: abs(x['importance']), reverse=True)[:8]
        
        # Separate positive and negative contributions
        positive_features = []
        negative_features = []
        
        for feat in top_features:
            feat_name = feat['feature'].replace('num__', '').replace('cat__', '')
            # Clean up name
            for key in self.feature_descriptions:
                if key.lower() in feat_name.lower():
                    feat_name = key
                    break
            
            if feat['importance'] > 0:
                positive_features.append({
                    'name': feat_name,
                    'value': feat['importance'],
                    'description': self.feature_descriptions.get(feat_name, feat_name)
                })
            else:
                negative_features.append({
                    'name': feat_name,
                    'value': abs(feat['importance']),
                    'description': self.feature_descriptions.get(feat_name, feat_name)
                })
        
        # Create diverging bar chart
        fig = go.Figure()
        
        if negative_features:
            fig.add_trace(go.Bar(
                name='Decreasing Risk',
                y=[f['name'] for f in negative_features],
                x=[-f['value'] for f in negative_features],
                orientation='h',
                marker_color='#28a745',
                text=[f"{f['value']:.3f}" for f in negative_features],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>%{text}<br><i>Decreases default risk</i><extra></extra>'
            ))
        
        if positive_features:
            fig.add_trace(go.Bar(
                name='Increasing Risk',
                y=[f['name'] for f in positive_features],
                x=[f['value'] for f in positive_features],
                orientation='h',
                marker_color='#dc3545',
                text=[f"{f['value']:.3f}" for f in positive_features],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>%{text}<br><i>Increases default risk</i><extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Risk Factors Analysis (Prediction: {prediction_prob:.1%} vs Base: {base_rate:.1%})",
            xaxis_title="Feature Importance",
            barmode='overlay',
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def generate_explanation_text(self, feature_importance: List[Dict[str, Any]], 
                                  risk_band: str, 
                                  prediction_prob: float) -> str:
        """Generate human-readable explanation"""
        if not feature_importance:
            return "Model explanations not available for this prediction."
        
        top_3 = sorted(feature_importance, key=lambda x: abs(x['importance']), reverse=True)[:3]
        
        explanation = f"This borrower is classified as **{risk_band}** with a {prediction_prob:.1%} probability of default.\n\n"
        explanation += "**Key factors influencing this assessment:**\n\n"
        
        for i, feat in enumerate(top_3, 1):
            feat_name = feat['feature'].replace('num__', '').replace('cat__', '')
            # Clean up
            for key in self.feature_descriptions:
                if key.lower() in feat_name.lower():
                    feat_name = key
                    break
            
            direction = "increases" if feat['importance'] > 0 else "decreases"
            description = self.feature_descriptions.get(feat_name, feat_name)
            
            explanation += f"{i}. **{feat_name}** ({description}) - {direction} default risk\n"
        
        # Add recommendation
        explanation += "\n**Recommendation:** "
        if prediction_prob < 0.20:
            explanation += "✅ **APPROVE** - Low risk profile suitable for loan approval with standard terms."
        elif prediction_prob < 0.50:
            explanation += "⚠️ **REVIEW** - Moderate risk requires additional assessment. Consider risk-based pricing or smaller loan amount."
        elif prediction_prob < 0.75:
            explanation += "🔍 **MANUAL REVIEW** - High risk requires thorough manual review and potentially enhanced monitoring or guarantor requirements."
        else:
            explanation += "❌ **REJECT** - Very high risk profile. Loan approval not recommended under current circumstances."
        
        return explanation
    
    def create_regional_comparison(self, borrower_region: str, 
                                   borrower_occupation: str,
                                   borrower_prob: float) -> go.Figure:
        """Create comparison chart showing borrower vs regional/occupational averages"""
        # Simulated benchmark data (in real implementation, fetch from database)
        regional_benchmarks = {
            'Northern': 0.28,
            'Ashanti': 0.19,
            'Volta': 0.22,
            'Eastern': 0.21,
            'Western': 0.25,
            'Central': 0.23
        }
        
        occupational_benchmarks = {
            'Farmer': 0.26,
            'Trader': 0.21,
            'Artisan': 0.23,
            'Teacher': 0.12,
            'Civil Servant': 0.10,
            'Business Owner': 0.24
        }
        
        fig = go.Figure()
        
        # Add borrower probability
        fig.add_trace(go.Bar(
            name='This Borrower',
            x=['Borrower'],
            y=[borrower_prob],
            marker_color='#007bff',
            text=[f"{borrower_prob:.1%}"],
            textposition='outside'
        ))
        
        # Add regional average
        regional_avg = regional_benchmarks.get(borrower_region, 0.23)
        fig.add_trace(go.Bar(
            name=f'{borrower_region} Region Avg',
            x=[f'{borrower_region} Avg'],
            y=[regional_avg],
            marker_color='#ffc107',
            text=[f"{regional_avg:.1%}"],
            textposition='outside'
        ))
        
        # Add occupational average
        occ_avg = occupational_benchmarks.get(borrower_occupation, 0.23)
        fig.add_trace(go.Bar(
            name=f'{borrower_occupation} Avg',
            x=[f'{borrower_occupation} Avg'],
            y=[occ_avg],
            marker_color='#17a2b8',
            text=[f"{occ_avg:.1%}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Risk Comparison: Borrower vs Benchmarks",
            yaxis_title="Default Probability",
            yaxis_tickformat='.0%',
            height=400,
            showlegend=False,
            barmode='group'
        )
        
        # Add reference line at 50% threshold
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold (50%)")
        
        return fig
