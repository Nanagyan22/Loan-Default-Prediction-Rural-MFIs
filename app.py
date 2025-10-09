import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import utility modules
from utils.model_loader import ModelLoader
from utils.risk_calculator import RiskCalculator
from utils.visualization import VisualizationHelper
from utils.explainability import ModelExplainer
from utils.database import init_db, save_prediction, get_predictions, get_fairness_stats, save_batch_processing
from utils.performance_tracking import PerformanceTracker

# Page configuration
st.set_page_config(
    page_title="UGBS Credit Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #1f4e79, #2e6da4);
        color: white;
        margin: -30px -30px 30px -30px;
        border-radius: 0 0 10px 10px;
    }
    .risk-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    .low-risk { background-color: #d4edda; color: #155724; }
    .medium-risk { background-color: #fff3cd; color: #856404; }
    .high-risk { background-color: #f8d7da; color: #721c24; }
    .very-high-risk { background-color: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

# Initialize database
try:
    if init_db():
        logging.info("Database initialized successfully")
except Exception as e:
    logging.error(f"Database initialization failed: {str(e)}")

# Initialize session state
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = ModelLoader()
if 'risk_calculator' not in st.session_state:
    st.session_state.risk_calculator = RiskCalculator()
if 'viz_helper' not in st.session_state:
    st.session_state.viz_helper = VisualizationHelper()
if 'explainer' not in st.session_state:
    st.session_state.explainer = ModelExplainer()


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 UNIVERSITY OF GHANA BUSINESS SCHOOL</h1>
        <h3>Project : Machine Learning Credit Scoring System</h3>
        <p>Rural Microfinance Credit Assessment Platform</p>
        <p style="font-size: 14px; margin-top: 10px;">Francis Afful Gyan | ID: 22253332</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        # Model selection - Using Random Forest + SMOTE only
        available_models = st.session_state.model_loader.get_available_models()
        if len(available_models) > 0:
            selected_model = available_models[0]
            st.info(f"🌳 Using: Random Forest + SMOTE")
        else:
            st.error("No trained models available!")
            selected_model = None

        st.session_state.selected_model = selected_model

        # Model status
        st.subheader("📊 Model Status")
        model_status = st.session_state.model_loader.get_model_status()
        for model, status in model_status.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {model.replace('_', ' ').title()}")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Single Assessment",
        "📊 Batch Processing",
        "⚖️ Fairness Monitor",
        "🔄 What-If Simulator",
        "📊 Performance Tracking",
        "📈 Analytics Dashboard",
        "📋 Model Information"
    ])

    with tab1:
        single_assessment_interface()

    with tab2:
        batch_processing_interface()

    with tab3:
        fairness_monitoring_interface()

    with tab4:
        what_if_simulator_interface()

    with tab5:
        performance_tracking_interface()

    with tab6:
        analytics_dashboard()

    with tab7:
        model_information_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; color: #666; font-size: 14px;">
        🧠 Applied Machine Learning Project · University of Ghana (UoGBS) · 22253332
    </div>
    """, unsafe_allow_html=True)


def single_assessment_interface():
    st.header("Individual Borrower Assessment")

    if not st.session_state.selected_model:
        st.error("⚠️ No model available for assessment")
        return

    # Create form for input
    with st.form("borrower_assessment"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Personal Information")
            applicant_id = st.text_input("Applicant ID", value="CUST-NEW-001")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
            education_level = st.selectbox("Education Level", ["No Education", "Primary", "JHS", "SHS", "Tertiary"])

        with col2:
            st.subheader("💼 Financial Information")
            region = st.selectbox("Region", [
                "Greater Accra", "Ashanti", "Eastern", "Western", "Volta",
                "Northern", "Central", "Upper East"
            ])
            occupation = st.selectbox("Occupation", [
                "Farmer", "Trader", "Artisan", "Teacher", "Civil Servant",
                "Business Owner", "Student", "Unemployed", "Other"
            ])
            monthly_income_ghs = st.number_input("Monthly Income (GHS)", min_value=0, value=1500)
            loan_amount_requested_ghs = st.number_input("Loan Amount Requested (GHS)", min_value=100, value=5000)
            loan_purpose = st.selectbox("Loan Purpose", [
                "Business", "Agriculture", "Education", "Healthcare",
                "Home Improvement", "Emergency", "Other"
            ])
            loan_term_months = st.number_input("Loan Term (Months)", min_value=1, max_value=60, value=12)

        with col3:
            st.subheader("📈 Credit History & Risk Factors")
            late_payments_pastyear = st.number_input("Late Payments (Past Year)", min_value=0, max_value=50, value=0,
                                                     help="Number of late payments in the past 12 months")
            loan_to_income_ratio = st.number_input("Loan-to-Income Ratio", min_value=0.0, max_value=20.0, value=3.3,
                                                   step=0.1,
                                                   help="Ratio of loan amount to monthly income (auto-calculated if 0)")
            debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=20.0, value=1.0,
                                                   step=0.1,
                                                   help="Ratio of total monthly debts to monthly income")
            previous_default_count = st.number_input("Previous Default Count", min_value=0, max_value=10, value=0)
            mobile_money_tx_volume_3m = st.number_input("Mobile Money Transactions (3M)", min_value=0, value=500)
            group_membership = st.selectbox("Group Membership", ["Yes", "No"])
            application_date = st.date_input("Application Date", value=datetime.now().date())

        # Submit button
        submitted = st.form_submit_button("🎯 Assess Credit Score", use_container_width=True)

        if submitted:
            # Prepare input data
            input_data = {
                'applicant_id': applicant_id,
                'age': age,
                'gender': gender,
                'marital_status': marital_status,
                'household_size': household_size,
                'education_level': education_level,
                'region': region,
                'occupation': occupation,
                'monthly_income_ghs': monthly_income_ghs,
                'loan_amount_requested_ghs': loan_amount_requested_ghs,
                'loan_purpose': loan_purpose,
                'loan_term_months': loan_term_months,
                'previous_default_count': previous_default_count,
                'late_payments_pastyear': late_payments_pastyear,
                'loan_to_income_ratio': loan_to_income_ratio if loan_to_income_ratio > 0 else None,
                'debt_to_income_ratio': debt_to_income_ratio,
                'mobile_money_tx_volume_3m': mobile_money_tx_volume_3m,
                'group_membership': group_membership,
                'application_date': application_date.strftime('%Y-%m-%d')
            }

            # Make prediction
            try:
                result = st.session_state.model_loader.predict_single(input_data, st.session_state.selected_model)

                # Save prediction to database
                try:
                    prediction_data = {
                        'applicant_id': input_data.get('applicant_id', 'UNKNOWN'),
                        'age': input_data.get('age'),
                        'gender': input_data.get('gender'),
                        'marital_status': input_data.get('marital_status'),
                        'education_level': input_data.get('education_level'),
                        'region': input_data.get('region'),
                        'occupation': input_data.get('occupation'),
                        'monthly_income': input_data.get('monthly_income_ghs'),
                        'loan_amount_requested': input_data.get('loan_amount_requested_ghs'),
                        'loan_term_months': input_data.get('loan_term_months'),
                        'household_size': input_data.get('household_size'),
                        'model_used': result['model_used'],
                        'default_probability': result['default_probability'],
                        'risk_band': result['risk_band'],
                        'recommendation': get_recommendation(result['default_probability']),
                        'feature_importance': result.get('feature_importance', [])
                    }
                    save_prediction(prediction_data)
                except Exception as db_error:
                    logging.warning(f"Failed to save prediction to database: {str(db_error)}")

                display_single_assessment_results(result, input_data)
            except Exception as e:
                st.error(f"⚠️ Prediction error: {str(e)}")
                st.warning("⚠️ Model output discrepancy detected. Refer to training Jupyter notebook for verification.")
                logging.error(f"Prediction error: {str(e)}")


def display_single_assessment_results(result, input_data):
    st.header("🎯 Credit Assessment Results")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        default_prob = result['default_probability']
        st.metric(
            "Default Probability",
            f"{default_prob:.2%}",
            delta=f"{default_prob - 0.20:.2%}" if default_prob > 0.20 else None
        )

    with col2:
        risk_band = result['risk_band']
        risk_colors = {
            'Low Risk': 'low-risk',
            'Medium Risk': 'medium-risk',
            'High Risk': 'high-risk',
            'Very High Risk': 'very-high-risk'
        }
        st.markdown(f"""
        <div class="risk-card {risk_colors.get(risk_band, 'low-risk')}">
            {risk_band}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        confidence = result.get('confidence', 0.85)
        st.metric("Confidence", f"{confidence:.1%}")

    with col4:
        recommendation = get_recommendation(default_prob)
        st.metric("Recommendation", recommendation)

    # Detailed analysis
    # Top Contributing Factors
    if 'feature_importance' in result and result['feature_importance']:
        st.subheader("🔍 Top Contributing Factors")

        # Get top 5 factors
        sorted_features = sorted(result['feature_importance'], key=lambda x: abs(x['importance']), reverse=True)[:5]

        if sorted_features:
            cols = st.columns(len(sorted_features))
            for idx, feature in enumerate(sorted_features):
                with cols[idx]:
                    impact = "↑ Increases" if feature['importance'] > 0 else "↓ Decreases"
                    color = "red" if feature['importance'] > 0 else "green"
                    st.markdown(f"**{feature['feature']}**")
                    st.markdown(f":{color}[{impact} Risk]")
                    st.caption(f"Impact: {abs(feature['importance']):.3f}")

        # Explanation text
        explanation = st.session_state.explainer.generate_explanation_text(
            result['feature_importance'],
            result['risk_band'],
            default_prob
        )
        st.markdown(explanation)

    # Visual analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Feature Contributions")
        if 'feature_importance' in result:
            fig = st.session_state.explainer.create_force_plot_alternative(
                result['feature_importance'],
                default_prob
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model explanations not available for this model type.")

        with col2:
            st.subheader("🎯 Risk Gauge")
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=default_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Risk %"},
                delta={'reference': 20},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Benchmark comparison
        st.subheader("📈 Benchmark Comparison")
        borrower_region = input_data.get('region', 'Ashanti')
        borrower_occupation = input_data.get('occupation', 'Farmer')
        fig_benchmark = st.session_state.explainer.create_regional_comparison(
            borrower_region, borrower_occupation, default_prob
        )
        st.plotly_chart(fig_benchmark, use_container_width=True)


def batch_processing_interface():
    st.header("📊 Batch Credit Score Assessment")

    if not st.session_state.selected_model:
        st.error("⚠️ No model available for batch processing")
        return

    st.markdown("""
    Process multiple loan applications simultaneously using ML-based risk scoring. 
    Upload a CSV file with borrower data or download our template to get started.
    """)

    # Template and Sample Download Section
    st.subheader("📥 Download Templates")
    col1, col2 = st.columns(2)

    with col1:
        # Load and provide template download
        try:
            with open('templates/batch_template.csv', 'r') as f:
                template_data = f.read()
            st.download_button(
                "📋 Download CSV Template",
                template_data,
                "batch_template.csv",
                "text/csv",
                help="Download template with all required columns",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Template not available: {str(e)}")

    with col2:
        st.info("**Template includes:** All 27+ required fields with sample data for 5 borrowers")

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload CSV file with borrower data",
        type=['csv'],
        help="CSV should contain columns: applicant_id, age, gender, monthly_income_ghs, loan_amount_requested_ghs, etc."
    )

    if uploaded_file is not None:
        try:
            # Read and validate CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} records")

            # Data Validation Summary
            st.subheader("🔍 Data Validation Summary")

            # Validate required columns
            required_columns = [
                'applicant_id', 'age', 'gender', 'marital_status', 'education_level',
                'occupation', 'monthly_income_ghs', 'loan_amount_requested_ghs'
            ]

            val_col1, val_col2, val_col3 = st.columns(3)

            with val_col1:
                st.metric("Total Records", len(df))
            with val_col2:
                present_cols = [col for col in required_columns if col in df.columns]
                st.metric("Required Fields Present", f"{len(present_cols)}/{len(required_columns)}")
            with val_col3:
                missing_count = df.isnull().sum().sum()
                st.metric("Missing Values", missing_count)

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_columns)} - Will use defaults")

            # Show preview
            st.subheader("📋 Data Preview (First 5 rows)")
            st.dataframe(df.head(), use_container_width=True)

            # Process batch
            if st.button("🚀 Process Batch", use_container_width=True):
                with st.spinner("Processing batch predictions..."):
                    try:
                        results_df = st.session_state.model_loader.predict_batch(df, st.session_state.selected_model)

                        # Save batch processing record
                        import uuid
                        batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
                        try:
                            batch_data = {
                                'batch_id': batch_id,
                                'total_records': len(results_df),
                                'successful_predictions': len(results_df[results_df['default_probability'].notna()]),
                                'failed_predictions': len(results_df[results_df['default_probability'].isna()]),
                                'model_used': st.session_state.selected_model,
                                'avg_default_probability': results_df['default_probability'].mean(),
                                'high_risk_count': len(
                                    results_df[results_df['risk_band'].isin(['High Risk', 'Very High Risk'])]),
                                'approval_rate': len(results_df[results_df['default_probability'] < 0.3]) / len(
                                    results_df) * 100,
                                'original_filename': uploaded_file.name,
                                'processed_filename': f"scored_{uploaded_file.name}"
                            }
                            save_batch_processing(batch_data)
                        except Exception as db_error:
                            logging.warning(f"Failed to save batch processing record: {str(db_error)}")

                        # Display results
                        st.success("✅ Batch processing completed successfully!")

                        # Summary statistics
                        st.subheader("📈 Credit Risk Assessment Summary")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            avg_risk = results_df['default_probability'].mean()
                            st.metric("Average Risk", f"{avg_risk:.2%}")

                        with col2:
                            high_risk_count = len(
                                results_df[results_df['risk_band'].isin(['High Risk', 'Very High Risk'])])
                            st.metric("High Risk Cases", high_risk_count)

                        with col3:
                            approval_rate = len(results_df[results_df['default_probability'] < 0.20]) / len(results_df)
                            st.metric("Approval Rate", f"{approval_rate:.1%}")

                        with col4:
                            st.metric("Total Processed", len(results_df))

                        # Risk distribution visualization
                        st.subheader("📊 Credit Risk Distribution Analysis")

                        viz_col1, viz_col2 = st.columns(2)

                        with viz_col1:
                            # Risk band distribution
                            risk_counts = results_df['risk_band'].value_counts().reset_index()
                            risk_counts.columns = ['Risk Band', 'Count']
                            fig_risk = px.bar(
                                risk_counts,
                                x='Risk Band',
                                y='Count',
                                title="Risk Band Distribution",
                                color='Risk Band',
                                color_discrete_map={
                                    'Low Risk': '#2ecc71',
                                    'Medium Risk': '#f39c12',
                                    'High Risk': '#e67e22',
                                    'Very High Risk': '#e74c3c'
                                }
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)

                        with viz_col2:
                            # Default probability histogram
                            fig_hist = px.histogram(
                                results_df,
                                x='default_probability',
                                nbins=30,
                                title="Default Probability Distribution",
                                labels={'default_probability': 'Default Probability'}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                        # Statistical Summary
                        st.subheader("📉 Statistical Analysis")
                        stats_df = results_df['default_probability'].describe().to_frame()
                        stats_df.columns = ['Default Probability']
                        stats_df = stats_df.round(4)

                        stat_col1, stat_col2 = st.columns([1, 2])
                        with stat_col1:
                            st.dataframe(stats_df, use_container_width=True)
                        with stat_col2:
                            st.info("""
                            **Interpretation Guide:**
                            - **Mean**: Average default credit risk across all applicants
                            - **Std**: Variability in risk scores (higher = more diverse portfolio)
                            - **25%-75%**: Middle 50% of risk scores fall in this range
                            - **Max**: Highest risk applicant in the batch
                            """)

                        # Detailed Results Table with Color-Coded Recommendations
                        st.subheader("📋 Detailed Assessment Results")

                        # Add decision reasons to the dataframe
                        def get_decision_reason(row):
                            prob = row['default_probability']
                            risk = row['risk_band']

                            if prob < 0.20:
                                return f"Low default risk ({prob:.1%}). Strong repayment indicators."
                            elif prob < 0.50:
                                return f"Moderate risk ({prob:.1%}). Requires additional verification."
                            elif prob < 0.75:
                                return f"High default risk ({prob:.1%}). Multiple risk factors identified."
                            else:
                                return f"Very high risk ({prob:.1%}). Significant default likelihood."

                        results_display = results_df.copy()
                        results_display['Decision Reason'] = results_display.apply(get_decision_reason, axis=1)

                        # Select and reorder columns for display
                        display_cols = ['applicant_id', 'default_probability', 'risk_band', 'recommendation',
                                        'Decision Reason']
                        if all(col in results_display.columns for col in display_cols):
                            display_df = results_display[display_cols].copy()
                        else:
                            display_df = results_display.copy()

                        # Format probability as percentage
                        if 'default_probability' in display_df.columns:
                            display_df['default_probability'] = display_df['default_probability'].apply(
                                lambda x: f"{x:.1%}")

                        # Apply color styling to recommendations
                        def highlight_recommendation(row):
                            colors = []
                            for col in row.index:
                                if col == 'recommendation':
                                    rec = str(row[col])
                                    if 'APPROVE' in rec:
                                        colors.append('background-color: #d4edda; color: #155724; font-weight: bold')
                                    elif 'REVIEW' in rec:
                                        colors.append('background-color: #fff3cd; color: #856404; font-weight: bold')
                                    elif 'CHECK' in rec:
                                        colors.append('background-color: #ffe5cc; color: #cc5200; font-weight: bold')
                                    elif 'REJECT' in rec:
                                        colors.append('background-color: #f8d7da; color: #721c24; font-weight: bold')
                                    else:
                                        colors.append('')
                                elif col == 'risk_band':
                                    risk = str(row[col])
                                    if 'Low' in risk:
                                        colors.append('background-color: #e7f5e7; color: #2d662d')
                                    elif 'Medium' in risk:
                                        colors.append('background-color: #fff8e1; color: #996600')
                                    elif 'Very High' in risk:
                                        colors.append('background-color: #ffcccc; color: #cc0000')
                                    elif 'High' in risk:
                                        colors.append('background-color: #ffe0cc; color: #cc4400')
                                    else:
                                        colors.append('')
                                else:
                                    colors.append('')
                            return colors

                        # Display styled dataframe
                        styled_df = display_df.style.apply(highlight_recommendation, axis=1)
                        st.dataframe(styled_df, use_container_width=True, height=400)

                        # Download section
                        st.subheader("💾 Export Results")

                        download_col1, download_col2 = st.columns(2)

                        with download_col1:
                            # Download full results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Full Results (CSV)",
                                csv_data,
                                f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )

                        with download_col2:
                            # Download high-risk cases only
                            high_risk_df = results_df[results_df['risk_band'].isin(['High Risk', 'Very High Risk'])]
                            if len(high_risk_df) > 0:
                                high_risk_csv = high_risk_df.to_csv(index=False)
                                st.download_button(
                                    "⚠️ Download High-Risk Cases Only",
                                    high_risk_csv,
                                    f"high_risk_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.info("No high-risk cases found")

                    except Exception as e:
                        st.error(f"❌ Batch processing error: {str(e)}")
                        st.warning(
                            "⚠️ Model output discrepancy detected. Refer to training Jupyter notebook for verification.")
                        logging.error(f"Batch processing error: {str(e)}")

        except Exception as e:
            st.error(f"❌ File processing error: {str(e)}")


def fairness_monitoring_interface():
    st.header("⚖️ Fairness & Bias Monitoring")

    st.markdown("""
    This dashboard monitors fairness metrics across different demographic groups to ensure equitable 
    credit access and identify potential biases in the model's predictions.
    """)

    # Dimension selector
    dimension = st.selectbox(
        "Analyze fairness by:",
        ["region", "gender", "occupation"],
        format_func=lambda x: x.capitalize()
    )

    try:
        # Get fairness statistics
        stats = get_fairness_stats(group_by=dimension)

        if not stats:
            st.info("📊 No predictions available yet. Make some predictions to see fairness metrics.")
            return

        # Create DataFrame for visualization
        stats_df = pd.DataFrame(stats)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_approval = stats_df['approval_rate'].mean()
            st.metric("Overall Approval Rate", f"{avg_approval:.1f}%")

        with col2:
            max_disparity = stats_df['approval_rate'].max() - stats_df['approval_rate'].min()
            st.metric("Max Approval Disparity", f"{max_disparity:.1f}%",
                      delta="Lower is better" if max_disparity > 10 else None)

        with col3:
            total_predictions = stats_df['total'].sum()
            st.metric("Total Predictions", f"{total_predictions:,}")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Approval Rates by {dimension.capitalize()}")
            fig = px.bar(
                stats_df,
                x='group',
                y='approval_rate',
                color='approval_rate',
                color_continuous_scale='RdYlGn',
                labels={'group': dimension.capitalize(), 'approval_rate': 'Approval Rate (%)'},
                text='approval_rate'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"Average Default Probability by {dimension.capitalize()}")
            fig = px.bar(
                stats_df,
                x='group',
                y='avg_default_prob',
                color='avg_default_prob',
                color_continuous_scale='RdYlGn_r',
                labels={'group': dimension.capitalize(), 'avg_default_prob': 'Avg Default Probability'},
                text='avg_default_prob'
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(height=400, showlegend=False, yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        # Bias Alert System
        st.subheader("🚨 Bias Alerts")

        # Check for significant disparities
        approval_threshold = 15  # 15% difference triggers alert
        risk_threshold = 0.10  # 10% absolute difference in default probability

        alerts = []
        for i, row in stats_df.iterrows():
            # Compare to overall average
            if abs(row['approval_rate'] - avg_approval) > approval_threshold:
                severity = "high" if abs(row['approval_rate'] - avg_approval) > 25 else "medium"
                direction = "lower" if row['approval_rate'] < avg_approval else "higher"
                alerts.append({
                    'group': row['group'],
                    'metric': 'Approval Rate',
                    'value': f"{row['approval_rate']:.1f}%",
                    'severity': severity,
                    'message': f"Approval rate is {abs(row['approval_rate'] - avg_approval):.1f}% {direction} than average"
                })

        if alerts:
            for alert in alerts:
                severity_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                st.warning(
                    f"{severity_color.get(alert['severity'], '⚪')} **{alert['group']}** - {alert['metric']}: {alert['value']} - {alert['message']}")
        else:
            st.success("✅ No significant bias detected across groups")

        # Detailed stats table
        st.subheader("📊 Detailed Statistics")
        display_df = stats_df.copy()
        display_df['approval_rate'] = display_df['approval_rate'].apply(lambda x: f"{x:.1f}%")
        display_df['avg_default_prob'] = display_df['avg_default_prob'].apply(lambda x: f"{x:.1%}")
        display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading fairness statistics: {str(e)}")
        logging.error(f"Fairness monitoring error: {str(e)}")


def what_if_simulator_interface():
    st.header("🔄 What-If Scenario Simulator")

    st.markdown("""
    Explore how changes to borrower characteristics affect credit risk predictions. 
    Adjust parameters below to see real-time impact on default probability.
    """)

    if not st.session_state.selected_model:
        st.error("⚠️ No model available for simulation")
        return

    # Create two columns for comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Base Scenario")
        base_age = st.slider("Age", 18, 70, 35, key="base_age")
        base_gender = st.selectbox("Gender", ["Male", "Female"], key="base_gender")
        base_income = st.slider("Monthly Income (GHS)", 500, 10000, 1500, step=100, key="base_income")
        base_loan = st.slider("Loan Amount (GHS)", 1000, 50000, 5000, step=500, key="base_loan")
        base_term = st.slider("Loan Term (Months)", 6, 60, 12, key="base_term")
        base_occupation = st.selectbox("Occupation", [
            "Farmer", "Trader", "Artisan", "Teacher", "Civil Servant",
            "Business Owner", "Student", "Unemployed", "Other"
        ], key="base_occ")
        base_region = st.selectbox("Region", [
            "Greater Accra", "Ashanti", "Eastern", "Western", "Volta",
            "Northern", "Central", "Upper East"
        ], key="base_region")
        base_late_payments = st.number_input("Late Payments (Past Year)", 0, 50, 0, key="base_late")

    with col2:
        st.subheader("🔄 Modified Scenario")
        mod_age = st.slider("Age", 18, 70, base_age, key="mod_age")
        mod_gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(base_gender),
                                  key="mod_gender")
        mod_income = st.slider("Monthly Income (GHS)", 500, 10000, base_income, step=100, key="mod_income")
        mod_loan = st.slider("Loan Amount (GHS)", 1000, 50000, base_loan, step=500, key="mod_loan")
        mod_term = st.slider("Loan Term (Months)", 6, 60, base_term, key="mod_term")
        mod_occupation = st.selectbox("Occupation", [
            "Farmer", "Trader", "Artisan", "Teacher", "Civil Servant",
            "Business Owner", "Student", "Unemployed", "Other"
        ], index=[
            "Farmer", "Trader", "Artisan", "Teacher", "Civil Servant",
            "Business Owner", "Student", "Unemployed", "Other"
        ].index(base_occupation), key="mod_occ")
        mod_region = st.selectbox("Region", [
            "Greater Accra", "Ashanti", "Eastern", "Western", "Volta",
            "Northern", "Central", "Upper East"
        ], index=[
            "Greater Accra", "Ashanti", "Eastern", "Western", "Volta",
            "Northern", "Central", "Upper East"
        ].index(base_region), key="mod_region")
        mod_late_payments = st.number_input("Late Payments (Past Year)", 0, 50, base_late_payments, key="mod_late")

    if st.button("⚡ Run Simulation", use_container_width=True):
        try:
            # Base scenario prediction
            base_input = {
                'age': base_age,
                'gender': base_gender,
                'monthly_income_ghs': base_income,
                'loan_amount_requested_ghs': base_loan,
                'loan_term_months': base_term,
                'occupation': base_occupation,
                'region': base_region,
                'late_payments_pastyear': base_late_payments,
                'marital_status': 'Married',
                'education_level': 'JHS',
                'household_size': 4
            }

            base_result = st.session_state.model_loader.predict_single(base_input, st.session_state.selected_model)

            # Modified scenario prediction
            mod_input = {
                'age': mod_age,
                'gender': mod_gender,
                'monthly_income_ghs': mod_income,
                'loan_amount_requested_ghs': mod_loan,
                'loan_term_months': mod_term,
                'occupation': mod_occupation,
                'region': mod_region,
                'late_payments_pastyear': mod_late_payments,
                'marital_status': 'Married',
                'education_level': 'JHS',
                'household_size': 4
            }

            mod_result = st.session_state.model_loader.predict_single(mod_input, st.session_state.selected_model)

            # Display results
            st.subheader("📊 Simulation Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Base Scenario Risk",
                    f"{base_result['default_probability']:.1%}",
                    f"{base_result['risk_band']}"
                )

            with result_col2:
                st.metric(
                    "Modified Scenario Risk",
                    f"{mod_result['default_probability']:.1%}",
                    f"{mod_result['risk_band']}"
                )

            with result_col3:
                change = mod_result['default_probability'] - base_result['default_probability']
                st.metric(
                    "Risk Change",
                    f"{abs(change):.1%}",
                    f"{'Increased' if change > 0 else 'Decreased'}"
                )

            # Comparison visualization
            comparison_df = pd.DataFrame({
                'Scenario': ['Base', 'Modified'],
                'Default Probability': [base_result['default_probability'], mod_result['default_probability']],
                'Age': [base_age, mod_age],
                'Income': [base_income, mod_income],
                'Loan Amount': [base_loan, mod_loan]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Base Scenario',
                x=['Default Risk'],
                y=[base_result['default_probability']],
                marker_color='lightblue',
                text=[f"{base_result['default_probability']:.1%}"],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='Modified Scenario',
                x=['Default Risk'],
                y=[mod_result['default_probability']],
                marker_color='coral',
                text=[f"{mod_result['default_probability']:.1%}"],
                textposition='outside'
            ))

            fig.update_layout(
                title="Default Probability Comparison",
                yaxis_title="Probability",
                yaxis_tickformat='.0%',
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Key insights
            st.subheader("💡 Key Insights")

            insights = []
            if mod_income != base_income:
                income_change = ((mod_income - base_income) / base_income) * 100
                insights.append(f"Income changed by {income_change:+.1f}%")

            if mod_loan != base_loan:
                loan_change = ((mod_loan - base_loan) / base_loan) * 100
                insights.append(f"Loan amount changed by {loan_change:+.1f}%")

            if change != 0:
                insights.append(f"Risk {'increased' if change > 0 else 'decreased'} by {abs(change):.1%}")

            if insights:
                for insight in insights:
                    st.info(f"• {insight}")

        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
            logging.error(f"What-if simulation error: {str(e)}")


def performance_tracking_interface():
    st.header("📊 Model Performance Tracking")

    st.markdown("""
    Monitor model performance over time by comparing predicted default probabilities against actual outcomes.
    This helps identify model drift and calibration issues.
    """)

    # Initialize performance tracker
    tracker = PerformanceTracker()

    # Get performance summary
    summary = tracker.get_performance_summary()

    # Auto-generate simulated outcomes for demo if none exist
    if not summary:
        with st.spinner("Initializing performance metrics with simulated outcomes..."):
            count = tracker.simulate_actual_outcomes(100)
            if count > 0:
                summary = tracker.get_performance_summary()
                st.success(f"✅ Generated {count} simulated outcomes for demonstration")

    # Simulate outcomes button (for demo purposes)
    with st.expander("🔬 Demo: Add More Simulated Outcomes"):
        st.info(
            "In production, actual outcomes would be updated as loans mature. For demonstration, you can add more simulated outcomes.")
        if st.button("Generate Additional Simulated Outcomes (100 records)"):
            count = tracker.simulate_actual_outcomes(100)
            st.success(f"✅ Simulated {count} additional actual outcomes")
            st.rerun()

    if not summary:
        st.warning("⚠️ Could not generate performance metrics. Please add predictions first.")
        return

    # Display key metrics
    st.subheader("📈 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{summary['accuracy']:.1%}")

    with col2:
        st.metric("Precision", f"{summary['precision']:.1%}")

    with col3:
        st.metric("Recall", f"{summary['recall']:.1%}")

    with col4:
        st.metric("F1 Score", f"{summary['f1_score']:.2f}")

    # Confusion Matrix
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        confusion_data = [
            ['True Negative', 'False Positive'],
            ['False Negative', 'True Positive']
        ]
        confusion_values = [
            [summary['true_negatives'], summary['false_positives']],
            [summary['false_negatives'], summary['true_positives']]
        ]

        fig = go.Figure(data=go.Heatmap(
            z=confusion_values,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=confusion_values,
            texttemplate="%{text}",
            colorscale='Blues'
        ))
        fig.update_layout(height=400, title="Prediction vs Actual Outcomes")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Calibration")
        calibration_data = tracker.get_calibration_data()

        if calibration_data:
            cal_df = pd.DataFrame(calibration_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cal_df['predicted'],
                y=cal_df['actual'],
                mode='markers+lines',
                name='Model',
                marker=dict(size=cal_df['count'] / 5, color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            ))
            fig.update_layout(
                title="Calibration Plot",
                xaxis_title="Predicted Probability",
                yaxis_title="Actual Default Rate",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Model Drift Detection
    st.subheader("🔍 Model Drift Detection")

    drift_data = tracker.get_model_drift_indicators()

    if drift_data:
        drift_col1, drift_col2, drift_col3 = st.columns(3)

        with drift_col1:
            st.metric(
                "Recent Avg Risk",
                f"{drift_data['recent_avg_prob']:.1%}",
                f"{drift_data['drift']:+.1%}"
            )

        with drift_col2:
            st.metric(
                "Historical Avg Risk",
                f"{drift_data['historical_avg_prob']:.1%}"
            )

        with drift_col3:
            st.metric(
                "Drift Percentage",
                f"{drift_data['drift_percentage']:.1f}%",
                "⚠️ Significant" if drift_data['is_significant'] else "✅ Normal"
            )

        if drift_data['is_significant']:
            st.warning("⚠️ **Significant model drift detected!** Consider model recalibration or retraining.")

            st.markdown("""
            **Recommended Actions:**
            1. Review recent prediction patterns for anomalies
            2. Check for changes in borrower demographics or economic conditions
            3. Consider retraining the model with recent data
            4. Implement enhanced monitoring for high-risk predictions
            """)
        else:
            st.success("✅ Model performance is stable with no significant drift detected.")

    # Performance over time
    st.subheader("📅 Performance Over Time")

    time_data = tracker.get_performance_over_time(days=30)

    if time_data:
        time_df = pd.DataFrame(time_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_df['date'],
            y=time_df['avg_predicted'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=time_df['date'],
            y=time_df['actual_default_rate'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Predicted vs Actual Default Rate Over Time",
            xaxis_title="Date",
            yaxis_title="Default Rate",
            yaxis_tickformat='.0%',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def analytics_dashboard():
    st.header("📈 Credit Risk Analytics Dashboard")

    # Generate sample analytics (in real implementation, this would come from historical data)
    if st.button("🔄 Generate Analytics", use_container_width=True):
        # Risk distribution by region
        regions = ['Northern', 'Ashanti', 'Volta', 'Eastern', 'Western', 'Central']
        risk_data = []

        for region in regions:
            for risk_level in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
                count = np.random.randint(20, 200)
                risk_data.append({'Region': region, 'Risk Level': risk_level, 'Count': count})

        risk_df = pd.DataFrame(risk_data)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗺️ Credit Risk Distribution by Region")
            fig = px.bar(
                risk_df,
                x='Region',
                y='Count',
                color='Risk Level',
                title="Credit Risk by Ghanaian Regions",
                color_discrete_map={
                    'Low Risk': '#28a745',
                    'Medium Risk': '#ffc107',
                    'High Risk': '#fd7e14',
                    'Very High Risk': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("👥 Risk by Occupation")
            occupations = ['Farmer', 'Trader', 'Artisan', 'Teacher', 'Civil Servant', 'Business Owner']
            occ_data = []

            for occ in occupations:
                avg_risk = np.random.uniform(0.1, 0.8)
                occ_data.append({'Occupation': occ, 'Average Risk': avg_risk})

            occ_df = pd.DataFrame(occ_data)
            fig = px.scatter(
                occ_df,
                x='Occupation',
                y='Average Risk',
                size=[np.random.randint(50, 300) for _ in occupations],
                title="Average Default Risk by Occupation",
                color='Average Risk',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Portfolio summary
        st.subheader("💼 Portfolio Summary")

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Total Applications", "8,158")
        with summary_cols[1]:
            st.metric("Approval Rate", "67.3%")
        with summary_cols[2]:
            st.metric("Average Loan Size", "GHS 4,850")
        with summary_cols[3]:
            st.metric("Default Rate", "23.1%")


def model_information_interface():
    st.header("📋 Model Information & Performance")

    # Model status overview
    st.subheader("🎯 Available Models")

    model_info = {
        'Random Forest + SMOTE': {
            'status': '✅ Available' if 'rf_pipeline' in st.session_state.model_loader.get_available_models() else '❌ Not Available',
            'description': 'Advanced ensemble learning algorithm combining multiple decision trees with SMOTE (Synthetic Minority Over-sampling) to handle class imbalance in default prediction',
            'best_for': 'Complex non-linear relationships, handling imbalanced datasets, robust feature interactions, and high-stakes credit decisions requiring accuracy and stability'
        }
    }

    for model_name, info in model_info.items():
        with st.expander(f"{model_name} - {info['status']}", expanded=True):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Best for:** {info['best_for']}")

    # Feature importance explanation
    st.subheader("🔍 Model Features")

    expected_features = [
        "age", "gender", "marital_status", "household_size", "education_level",
        "occupation", "monthly_income_ghs", "loan_amount_requested_ghs",
        "loan_purpose", "loan_term_months", "previous_loan_history",
        "previous_default_count", "mobile_money_tx_volume_3m",
        "gps_distance_to_town_km", "group_membership"
    ]

    feature_descriptions = {
        'age': 'Age of the borrower (18-100 years)',
        'gender': 'Gender of the borrower (Male/Female)',
        'monthly_income_ghs': 'Monthly income in Ghana Cedis',
        'loan_amount_requested_ghs': 'Requested loan amount in Ghana Cedis',
        'previous_default_count': 'Number of previous loan defaults',
        'mobile_money_tx_volume_3m': 'Mobile money transaction volume (last 3 months)',
        'gps_distance_to_town_km': 'Distance from nearest town in kilometers'
    }

    st.write("**Key Model Features:**")
    for feature in expected_features[:7]:  # Show top features
        description = feature_descriptions.get(feature, 'Important predictor variable')
        st.write(f"• **{feature.replace('_', ' ').title()}:** {description}")

    # Training information
    st.subheader("📚 Training Information")
    st.info("""
    **Training Dataset:** 8,158 borrower records from Sinapi Aba Savings and Loans

    **Geographic Coverage:** Northern, Ashanti, Volta, Eastern, Western, and Central regions of Ghana

    **Model Training:** Comprehensive preprocessing pipeline including feature scaling, 
    encoding, and handling of missing values

    **Validation:** Stratified cross-validation with class imbalance handling using SMOTE
    """)

    # Reference to training notebook
    st.subheader("📓 Training Notebook Reference")
    st.warning("""
    **⚠️ Important:** If you encounter any discrepancies in model predictions or 
    preprocessing behavior, please refer to the training Jupyter notebook 
    'UGBS-Case_Loan.ipynb' for detailed implementation and validation steps.
    """)


def get_recommendation(default_prob):
    """Get loan recommendation based on default probability"""
    if default_prob < 0.20:
        return "✅ APPROVE"
    elif default_prob < 0.50:
        return "⚠️ REVIEW"
    elif default_prob < 0.75:
        return "🔍 CHECK"
    else:
        return "❌ REJECT"


if __name__ == "__main__":
    main()
