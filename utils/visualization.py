import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import streamlit as st

class VisualizationHelper:
    """Helper class for creating visualizations"""
    
    def __init__(self):
        self.color_palette = {
            'low_risk': '#28a745',
            'medium_risk': '#ffc107',
            'high_risk': '#fd7e14', 
            'very_high_risk': '#dc3545',
            'primary': '#007bff',
            'secondary': '#6c757d'
        }
    
    def create_risk_gauge(self, probability: float, title: str = "Default Risk") -> go.Figure:
        """Create a gauge chart for risk probability"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 20},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_palette['primary']},
                'steps': [
                    {'range': [0, 20], 'color': self.color_palette['low_risk']},
                    {'range': [20, 50], 'color': self.color_palette['medium_risk']},
                    {'range': [50, 75], 'color': self.color_palette['high_risk']},
                    {'range': [75, 100], 'color': self.color_palette['very_high_risk']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_feature_importance_chart(self, importance_data: List[Dict[str, Any]], chart_type: str = "bar") -> go.Figure:
        """Create feature importance visualization"""
        df = pd.DataFrame(importance_data)
        df = df.sort_values('importance', key=abs, ascending=True)
        
        if chart_type == "bar":
            fig = px.bar(
                df,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance",
                color='importance',
                color_continuous_scale='RdYlBu_r'
            )
        else:
            # Horizontal bar with custom colors
            colors = [self.color_palette['very_high_risk'] if x < 0 else self.color_palette['low_risk'] 
                     for x in df['importance']]
            
            fig = go.Figure(go.Bar(
                x=df['importance'],
                y=df['feature'],
                orientation='h',
                marker_color=colors
            ))
            
        fig.update_layout(
            height=400,
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        
        return fig
    
    def create_risk_distribution_pie(self, risk_counts: Dict[str, int]) -> go.Figure:
        """Create pie chart for risk distribution"""
        labels = list(risk_counts.keys())
        values = list(risk_counts.values())
        colors = [
            self.color_palette['low_risk'],
            self.color_palette['medium_risk'], 
            self.color_palette['high_risk'],
            self.color_palette['very_high_risk']
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=colors[:len(labels)]
        )])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title_text="Risk Distribution",
            height=400
        )
        
        return fig
    
    def create_regional_heatmap(self, regional_data: pd.DataFrame) -> go.Figure:
        """Create heatmap for regional risk analysis"""
        # Pivot data for heatmap
        pivot_data = regional_data.pivot(index='Region', columns='Risk Level', values='Count')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn_r',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Regional Risk Heatmap",
            xaxis_title="Risk Level",
            yaxis_title="Region",
            height=500
        )
        
        return fig
    
    def create_portfolio_metrics_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create dashboard with multiple portfolio metrics"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Risk', 'Risk Distribution', 'Expected Defaults', 'Risk Score'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Average Risk Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['average_risk'] * 100,
            domain={'row': 0, 'column': 0},
            title={'text': "Avg Risk %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': self.color_palette['primary']},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}]}
        ), row=1, col=1)
        
        # Risk Distribution Bar
        risk_dist = metrics.get('risk_distribution', {})
        fig.add_trace(go.Bar(
            x=list(risk_dist.keys()),
            y=list(risk_dist.values()),
            marker_color=[self.color_palette['low_risk'], 
                         self.color_palette['medium_risk'],
                         self.color_palette['high_risk'],
                         self.color_palette['very_high_risk']][:len(risk_dist)]
        ), row=1, col=2)
        
        # Expected Defaults
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('expected_defaults', 0),
            title={'text': "Expected Defaults"},
            number={'suffix': " loans"}
        ), row=2, col=1)
        
        # Portfolio Risk Score
        fig.add_trace(go.Indicator(
            mode="delta",
            value=1,
            delta={'reference': 0.5},
            title={'text': metrics.get('risk_score', 'Moderate')},
        ), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Portfolio Risk Dashboard")
        return fig
    
    def create_probability_distribution(self, probabilities: List[float]) -> go.Figure:
        """Create histogram of probability distribution"""
        fig = go.Figure(data=[go.Histogram(
            x=probabilities,
            nbinsx=20,
            marker_color=self.color_palette['primary'],
            opacity=0.7
        )])
        
        # Add risk threshold lines
        fig.add_vline(x=0.2, line_dash="dash", line_color=self.color_palette['low_risk'], 
                      annotation_text="Low Risk Threshold")
        fig.add_vline(x=0.5, line_dash="dash", line_color=self.color_palette['medium_risk'],
                      annotation_text="Medium Risk Threshold")
        fig.add_vline(x=0.75, line_dash="dash", line_color=self.color_palette['high_risk'],
                      annotation_text="High Risk Threshold")
        
        fig.update_layout(
            title="Default Probability Distribution",
            xaxis_title="Default Probability",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def create_comparison_chart(self, data: Dict[str, List[float]], title: str = "Model Comparison") -> go.Figure:
        """Create comparison chart for multiple models"""
        fig = go.Figure()
        
        colors = [self.color_palette['primary'], self.color_palette['secondary']]
        
        for i, (model_name, values) in enumerate(data.items()):
            fig.add_trace(go.Box(
                y=values,
                name=model_name,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Default Probability",
            height=400
        )
        
        return fig
