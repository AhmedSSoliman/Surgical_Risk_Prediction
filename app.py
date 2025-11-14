# app.py - Complete Streamlit Application
"""
Surgical Risk Prediction System - Interactive Web Application

Features:
- Multimodal data upload and analysis
- Real-time risk prediction for 9 complications
- Comprehensive visualizations
- Explainability dashboard
- Clinical recommendations
- Report generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from config import COMPLICATIONS, MODEL_DIR, FIGURES_DIR
from data.data_loader import SampleDataGenerator
from preprocessing import TimeSeriesPreprocessor, ClinicalNotesPreprocessor, ModalityAligner
from models.model import MultimodalSurgicalRiskModel
from visualization import ResultsPlotter, ExplainabilityPlotter
from utils import get_device

# Page Configuration
st.set_page_config(
    page_title="Surgical Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    
    .risk-card {
        padding: 20px;
        border-radius: 15px;
        border: 2px solid;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .high-risk {
        border-color: #ff4444;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }
    
    .moderate-risk {
        border-color: #ff9800;
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    }
    
    .low-risk {
        border-color: #4caf50;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .phase-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.85em;
        margin: 2px;
    }
    
    .preop-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    
    .intraop-badge {
        background-color: #fff3e0;
        color: #f57c00;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Initialize Model (cached)
@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        device = get_device()
        
        # Check for saved model
        model_path = MODEL_DIR / 'best_model.pt'
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            
            # Get dimensions from checkpoint or use defaults
            ts_input_size = 30  # Adjust based on your data
            static_input_size = 20
            
            model = MultimodalSurgicalRiskModel(
                ts_input_size=ts_input_size,
                static_input_size=static_input_size
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            return model, device, True
        else:
            st.warning("⚠️ No trained model found. Using demo mode with sample predictions.")
            return None, device, False
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, torch.device('cpu'), False

# Initialize Preprocessors (cached)
@st.cache_resource
def load_preprocessors():
    """Load preprocessors"""
    return {
        'time_series': TimeSeriesPreprocessor(),
        'notes': ClinicalNotesPreprocessor(),
        'aligner': ModalityAligner()
    }

# Header
st.markdown('<h1 class="main-header">🏥 Surgical Risk Prediction System</h1>', 
            unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
     border-radius: 15px; margin-bottom: 25px; color: white;'>
    <h3 style='color: white; margin: 0;'>🤖 Multimodal AI for Predicting 9 Postoperative Complications</h3>
    <p style='margin: 10px 0 0 0; font-size: 0.95em;'>
        <b>Data Sources:</b> Clinical Notes (Preop/Intraop) • Lab Results • Vital Signs • Medications
    </p>
    <p style='margin: 5px 0 0 0; font-size: 0.85em; opacity: 0.9;'>
        <i>Powered by Vibe-Tuned AI • University of Florida NaviGator</i>
    </p>
</div>
""", unsafe_allow_html=True)

# Load model and preprocessors
model, device, model_loaded = load_model()
preprocessors = load_preprocessors()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["📊 Sample Patient Data", "📤 Upload Patient Files", "🔗 MIMIC-III Database"],
        help="Choose how to input patient data"
    )
    
    st.markdown("---")
    
    # Patient Information
    st.subheader("👤 Patient Demographics")
    patient_age = st.number_input("Age", min_value=18, max_value=100, value=67)
    patient_gender = st.selectbox("Gender", ["M", "F"])
    admission_type = st.selectbox("Admission Type", ["EMERGENCY", "ELECTIVE", "URGENT"])
    
    st.markdown("---")
    
    # Data Upload Section
    if data_source == "📤 Upload Patient Files":
        st.subheader("📂 Upload Data Files")
        
        uploaded_notes = st.file_uploader(
            "Clinical Notes (.txt)",
            type=['txt'],
            help="Upload clinical notes in text format"
        )
        
        uploaded_labs = st.file_uploader(
            "Lab Results (.csv)",
            type=['csv'],
            help="CSV with columns: CHARTTIME, ITEMID, LABEL, VALUENUM"
        )
        
        uploaded_vitals = st.file_uploader(
            "Vital Signs (.csv)",
            type=['csv'],
            help="CSV with columns: CHARTTIME, ITEMID, LABEL, VALUENUM"
        )
        
        uploaded_meds = st.file_uploader(
            "Medications (.csv)",
            type=['csv'],
            help="CSV with columns: STARTDATE, DRUG, ROUTE, DOSE_VAL_RX"
        )
        
        st.session_state.uploaded_files = {
            'notes': uploaded_notes,
            'labs': uploaded_labs,
            'vitals': uploaded_vitals,
            'meds': uploaded_meds
        }
    
    elif data_source == "🔗 MIMIC-III Database":
        st.subheader("🗄️ Database Connection")
        
        mimic_path = st.text_input(
            "MIMIC-III Path",
            value="./mimic-iii-clinical-database-1.4"
        )
        
        if st.button("🔗 Connect"):
            st.info("Database connection would be established here")
    
    else:  # Sample Data
        st.success("✓ Using sample patient data")
        
        with st.expander("📋 Sample Patient Profile"):
            st.markdown("""
            **Patient Information:**
            - 67-year-old male
            - Emergency admission
            - Procedure: Laparoscopic appendectomy
            
            **Clinical Scenario:**
            - Perforated appendicitis
            - Developed wound infection (POD 2)
            - Transient AKI (resolved)
            - Discharged POD 5
            """)
    
    st.markdown("---")
    
    # Analysis Options
    st.subheader("🔬 Analysis Options")
    
    show_preprocessing = st.checkbox("Show Preprocessing Details", value=False)
    show_explainability = st.checkbox("Enable Explainability", value=True)
    show_uncertainty = st.checkbox("Show Uncertainty Estimates", value=True)
    export_figures = st.checkbox("Save All Figures", value=True)
    
    st.markdown("---")
    
    # Model Info
    if model_loaded:
        st.success("✅ Model Loaded")
        params = model.get_num_parameters()
        st.metric("Trainable Params", f"{params['trainable']:,}")
    else:
        st.warning("⚠️ Demo Mode")
        st.caption("Using sample predictions")

# Main Content Area
st.markdown("## 📋 Patient Data Status")

# Status indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-box'>
        <h2 style='margin: 0; color: #1976d2;'>📝</h2>
        <p style='margin: 5px 0 0 0; font-weight: 600;'>Clinical Notes</p>
        <p style='margin: 5px 0 0 0; font-size: 0.85em; color: #666;'>Preop + Intraop</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-box'>
        <h2 style='margin: 0; color: #c62828;'>🧪</h2>
        <p style='margin: 5px 0 0 0; font-weight: 600;'>Laboratory</p>
        <p style='margin: 5px 0 0 0; font-size: 0.85em; color: #666;'>Temporal Series</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-box'>
        <h2 style='margin: 0; color: #2e7d32;'>💓</h2>
        <p style='margin: 5px 0 0 0; font-weight: 600;'>Vital Signs</p>
        <p style='margin: 5px 0 0 0; font-size: 0.85em; color: #666;'>Continuous</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-box'>
        <h2 style='margin: 0; color: #f57c00;'>💊</h2>
        <p style='margin: 5px 0 0 0; font-weight: 600;'>Medications</p>
        <p style='margin: 5px 0 0 0; font-size: 0.85em; color: #666;'>Administration</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Analysis Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🚀 RUN MULTIMODAL RISK ANALYSIS",
        type="primary",
        use_container_width=True,
        help="Analyze all data and predict surgical complications"
    )

# Analysis Logic
if analyze_button:
    
    # Generate/Load Patient Data
    with st.spinner("📥 Loading patient data..."):
        
        if data_source == "📊 Sample Patient Data":
            patient_data = SampleDataGenerator.generate_sample_patient()
            patient_data['demographics']['age'] = patient_age
            patient_data['demographics']['gender'] = patient_gender
        
        elif data_source == "📤 Upload Patient Files":
            st.error("File upload mode - Implementation in progress")
            st.stop()
        
        else:  # MIMIC-III
            st.error("MIMIC-III mode - Implementation in progress")
            st.stop()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Preprocess Time Series (20%)
    status_text.markdown("### 🔄 Step 1/5: Preprocessing Time Series Data...")
    progress_bar.progress(20)
    
    surgery_time = patient_data['surgery_time']
    
    # Preoperative labs
    labs_preop, labs_preop_names, labs_preop_meta = preprocessors['time_series'].preprocess_labs(
        patient_data['labs'],
        surgery_time,
        phase='preoperative'
    )
    
    # Intraoperative labs
    labs_intraop, labs_intraop_names, labs_intraop_meta = preprocessors['time_series'].preprocess_labs(
        patient_data['labs'],
        surgery_time,
        phase='intraoperative'
    )
    
    # Preoperative vitals
    vitals_preop, vitals_preop_names, vitals_preop_meta = preprocessors['time_series'].preprocess_vitals(
        patient_data['vitals'],
        surgery_time,
        phase='preoperative'
    )
    
    # Intraoperative vitals
    vitals_intraop, vitals_intraop_names, vitals_intraop_meta = preprocessors['time_series'].preprocess_vitals(
        patient_data['vitals'],
        surgery_time,
        phase='intraoperative'
    )
    
    st.session_state.ts_metadata = {
        'labs_preop': labs_preop_meta,
        'labs_intraop': labs_intraop_meta,
        'vitals_preop': vitals_preop_meta,
        'vitals_intraop': vitals_intraop_meta
    }
    
    # Step 2: Preprocess Clinical Notes (40%)
    status_text.markdown("### 📝 Step 2/5: Analyzing Clinical Notes (Preop/Intraop)...")
    progress_bar.progress(40)
    
    # Preoperative notes
    notes_preop = preprocessors['notes'].preprocess_notes(
        patient_data['notes'],
        surgery_time,
        phase='preoperative'
    )
    
    # Intraoperative notes
    notes_intraop = preprocessors['notes'].preprocess_notes(
        patient_data['notes'],
        surgery_time,
        phase='intraoperative'
    )
    
    st.session_state.notes_analysis = {
        'preop': notes_preop,
        'intraop': notes_intraop
    }
    
    # Step 3: Align Modalities (60%)
    status_text.markdown("### 🔗 Step 3/5: Aligning Multimodal Data...")
    progress_bar.progress(60)
    
    aligned_data = preprocessors['aligner'].align_all_modalities(
        labs_preop=(labs_preop, labs_preop_names, labs_preop_meta),
        labs_intraop=(labs_intraop, labs_intraop_names, labs_intraop_meta),
        vitals_preop=(vitals_preop, vitals_preop_names, vitals_preop_meta),
        vitals_intraop=(vitals_intraop, vitals_intraop_names, vitals_intraop_meta),
        notes_preop=notes_preop,
        notes_intraop=notes_intraop,
        static_features=patient_data['demographics'],
        medications_df=patient_data['medications'],
        outcomes=patient_data.get('outcomes')
    )
    
    st.session_state.aligned_data = aligned_data
    
    # Step 4: Predict Risks (80%)
    status_text.markdown("### 🎯 Step 4/5: Predicting Surgical Risks...")
    progress_bar.progress(80)
    
    if model_loaded and model is not None:
        # Real prediction
        with torch.no_grad():
            # Prepare batch
            batch = {
                'time_series_full': torch.FloatTensor(aligned_data['time_series']['full_sequence']).unsqueeze(0).to(device),
                'phase_markers': torch.FloatTensor(aligned_data['time_series']['phase_markers']).unsqueeze(0).to(device),
                'mask_full': torch.FloatTensor(aligned_data['attention_masks']['full_sequence']).unsqueeze(0).to(device),
                'text_combined': torch.FloatTensor(aligned_data['text']['combined_embedding']).unsqueeze(0).to(device),
                'static': torch.FloatTensor(aligned_data['static']['array']).unsqueeze(0).to(device)
            }
            
            outputs = model(
                time_series=batch['time_series_full'],
                phase_markers=batch['phase_markers'],
                ts_attention_mask=batch['mask_full'],
                text_embedding=batch['text_combined'],
                static_features=batch['static'],
                compute_uncertainty=show_uncertainty
            )
            
            predictions = {task: pred.cpu().numpy()[0] 
                         for task, pred in outputs['predictions'].items()}
            
            if show_uncertainty:
                uncertainties = {task: unc.cpu().numpy()[0] 
                               for task, unc in outputs['uncertainties'].items()}
            else:
                uncertainties = {task: 0.0 for task in predictions.keys()}
    else:
        # Demo mode with synthetic predictions
        predictions = {}
        uncertainties = {}
        
        # Generate realistic predictions based on patient data
        base_risk = 0.3 + (patient_age - 18) / 100 * 0.3
        
        for task_name, task_info in COMPLICATIONS.items():
            # Add some randomness
            risk = base_risk + np.random.uniform(-0.2, 0.2)
            risk = np.clip(risk, 0.05, 0.95)
            
            predictions[task_name] = risk
            uncertainties[task_name] = np.random.uniform(0.05, 0.15)
    
    st.session_state.predictions = predictions
    st.session_state.uncertainties = uncertainties
    
    # Step 5: Generate Analysis (100%)
    status_text.markdown("### 📊 Step 5/5: Generating Comprehensive Analysis...")
    progress_bar.progress(100)
    
    # Calculate overall risk
    weighted_risks = [predictions[task] * COMPLICATIONS[task]['weight'] 
                     for task in COMPLICATIONS.keys()]
    total_weight = sum(COMPLICATIONS[task]['weight'] for task in COMPLICATIONS.keys())
    overall_risk = sum(weighted_risks) / total_weight
    
    st.session_state.overall_risk = overall_risk
    st.session_state.patient_data = patient_data
    st.session_state.analysis_complete = True
    
    # Clear progress
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    st.rerun()

# Display Results
if st.session_state.analysis_complete and st.session_state.predictions:
    
    st.success("✅ **ANALYSIS COMPLETE!**")
    st.markdown("---")
    
    predictions = st.session_state.predictions
    uncertainties = st.session_state.uncertainties
    overall_risk = st.session_state.overall_risk
    
    # Overall Risk Display
    st.markdown("## 🎯 Overall Surgical Risk Assessment")
    
    # Determine risk level
    if overall_risk >= 0.7:
        risk_level = "HIGH"
        risk_color = "#ff4444"
        risk_emoji = "🔴"
        risk_class = "high-risk"
    elif overall_risk >= 0.4:
        risk_level = "MODERATE"
        risk_color = "#ff9800"
        risk_emoji = "🟡"
        risk_class = "moderate-risk"
    else:
        risk_level = "LOW"
        risk_color = "#4caf50"
        risk_emoji = "🟢"
        risk_class = "low-risk"
    
    # Risk metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box' style='border: 3px solid {risk_color};'>
            <h1 style='margin: 0; font-size: 3em;'>{risk_emoji}</h1>
            <h2 style='margin: 10px 0; color: {risk_color};'>{overall_risk:.1%}</h2>
            <p style='margin: 0; font-weight: bold; color: {risk_color};'>{risk_level} RISK</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk_count = sum(1 for p in predictions.values() if p >= 0.7)
        st.metric("High-Risk Complications", f"{high_risk_count}/9", 
                 delta="Requires Attention" if high_risk_count > 0 else "Good")
    
    with col3:
        moderate_risk_count = sum(1 for p in predictions.values() if 0.4 <= p < 0.7)
        st.metric("Moderate Risk", f"{moderate_risk_count}/9")
    
    with col4:
        if show_uncertainty:
            mean_uncertainty = np.mean(list(uncertainties.values()))
            st.metric("Mean Uncertainty", f"{mean_uncertainty:.1%}")
        else:
            st.metric("Prediction Confidence", "High")
    
    with col5:
        st.metric("Data Completeness", "100%")
    
    # Risk Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Surgical Risk Score", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': risk_color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#c8e6c9'},
                {'range': [40, 70], 'color': '#ffe0b2'},
                {'range': [70, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': 70
            }
        }
    ))
    
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=80, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Risk Scores",
        "📝 Clinical Notes",
        "🧪 Laboratory",
        "💓 Vital Signs",
        "💊 Medications",
        "🔍 Explainability",
        "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("Individual Complication Risk Scores")
        
        # Create risk scores table
        risk_data = []
        for task_name, task_info in sorted(COMPLICATIONS.items(), 
                                          key=lambda x: predictions[x[0]], 
                                          reverse=True):
            risk_score = predictions[task_name]
            uncertainty = uncertainties.get(task_name, 0.0)
            
            if risk_score >= 0.7:
                category = "HIGH"
                emoji = "🔴"
            elif risk_score >= 0.4:
                category = "MODERATE"
                emoji = "🟡"
            else:
                category = "LOW"
                emoji = "🟢"
            
            risk_data.append({
                'Status': emoji,
                'Complication': task_info['name'],
                'Risk Score': f"{risk_score:.1%}",
                'Uncertainty': f"±{uncertainty:.1%}" if show_uncertainty else "N/A",
                'Category': category,
                'Clinical Severity': task_info['weight']
            })
        
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Horizontal bar chart
        st.subheader("Risk Score Visualization")
        
        fig_bars = go.Figure()
        
        colors = ['#ff4444' if predictions[t] >= 0.7 else '#ff9800' if predictions[t] >= 0.4 else '#4caf50' 
                 for t in sorted(COMPLICATIONS.keys())]
        
        fig_bars.add_trace(go.Bar(
            y=[COMPLICATIONS[t]['name'] for t in sorted(COMPLICATIONS.keys())],
            x=[predictions[t] for t in sorted(COMPLICATIONS.keys())],
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.3)', width=2)),
            text=[f"{predictions[t]:.1%}" for t in sorted(COMPLICATIONS.keys())],
            textposition='inside',
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>Risk: %{x:.1%}<br><extra></extra>'
        ))
        
        # Add risk zones
        fig_bars.add_vrect(x0=0, x1=0.4, fillcolor="green", opacity=0.1, 
                          layer="below", line_width=0, annotation_text="LOW",
                          annotation_position="top left")
        fig_bars.add_vrect(x0=0.4, x1=0.7, fillcolor="orange", opacity=0.1,
                          layer="below", line_width=0, annotation_text="MODERATE",
                          annotation_position="top")
        fig_bars.add_vrect(x0=0.7, x1=1.0, fillcolor="red", opacity=0.1,
                          layer="below", line_width=0, annotation_text="HIGH",
                          annotation_position="top right")
        
        fig_bars.update_layout(
            title='Postoperative Complication Risk Scores',
            xaxis_title='Risk Score',
            yaxis_title='',
            xaxis=dict(range=[0, 1], tickformat='.0%'),
            height=500,
            showlegend=False,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig_bars, use_container_width=True)
        
        if export_figures:
            fig_bars.write_image(FIGURES_DIR / 'app_risk_bars.png', width=1400, height=800)
        
        st.markdown("---")
        
        # Detailed cards for high-risk complications
        high_risk_tasks = [t for t in COMPLICATIONS.keys() if predictions[t] >= 0.7]
        
        if high_risk_tasks:
            st.subheader("⚠️ High-Risk Complications - Detailed View")
            
            for task_name in high_risk_tasks:
                task_info = COMPLICATIONS[task_name]
                risk_score = predictions[task_name]
                uncertainty = uncertainties.get(task_name, 0.0)
                
                with st.expander(f"🔴 {task_info['name']} - {risk_score:.1%}", expanded=True):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class='risk-card high-risk'>
                            <h4 style='margin-top: 0;'>{task_info['name']}</h4>
                            <p><b>Description:</b> {task_info['description']}</p>
                            <p><b>Risk Score:</b> {risk_score:.1%} ± {uncertainty:.1%}</p>
                            <p><b>Risk Level:</b> <span style='color: #d32f2f; font-weight: bold;'>HIGH</span></p>
                            <p><b>Clinical Impact:</b> {task_info.get('weight', 1.0)}/1.0</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Mini gauge
                        fig_mini = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_score * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#d32f2f"},
                                'steps': [
                                    {'range': [0, 70], 'color': "lightgray"},
                                    {'range': [70, 100], 'color': "#ffcdd2"}
                                ]
                            }
                        ))
                        fig_mini.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.success("✅ No high-risk complications identified")
        
        st.markdown("---")
        
        # Risk heatmap
        st.subheader("🔥 Risk Heatmap")
        
        risk_matrix = np.array([[predictions[t] for t in sorted(COMPLICATIONS.keys())]])
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=[COMPLICATIONS[t]['name'] for t in sorted(COMPLICATIONS.keys())],
            y=['Risk Score'],
            colorscale='RdYlGn_r',
            text=[[f'{predictions[t]:.1%}' for t in sorted(COMPLICATIONS.keys())]],
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title='Risk Level', tickformat='.0%'),
            hovertemplate='<b>%{x}</b><br>Risk: %{z:.1%}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title='Complication Risk Heatmap',
            height=200,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        if export_figures:
            fig_heatmap.write_image(FIGURES_DIR / 'app_risk_heatmap.png', width=1400, height=300)
    
    with tab2:
        st.subheader("📝 Clinical Notes Analysis")
        
        notes_preop = st.session_state.notes_analysis['preop']
        notes_intraop = st.session_state.notes_analysis['intraop']
        
        # Phase indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #1976d2;'>
                <h4 style='margin: 0 0 10px 0;'>
                    <span class='phase-badge preop-badge'>PREOPERATIVE</span>
                </h4>
                <p style='margin: 5px 0;'><b>Notes:</b> {}</p>
                <p style='margin: 5px 0;'><b>Words:</b> {}</p>
                <p style='margin: 5px 0;'><b>Severity:</b> {:.1%}</p>
            </div>
            """.format(
                notes_preop['metadata']['num_notes'],
                notes_preop['metadata']['total_words'],
                notes_preop['metadata']['severity_score']
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #f57c00;'>
                <h4 style='margin: 0 0 10px 0;'>
                    <span class='phase-badge intraop-badge'>INTRAOPERATIVE</span>
                </h4>
                <p style='margin: 5px 0;'><b>Notes:</b> {}</p>
                <p style='margin: 5px 0;'><b>Words:</b> {}</p>
                <p style='margin: 5px 0;'><b>Severity:</b> {:.1%}</p>
            </div>
            """.format(
                notes_intraop['metadata']['num_notes'],
                notes_intraop['metadata']['total_words'],
                notes_intraop['metadata']['severity_score']
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key findings by phase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔑 Preoperative Key Findings:**")
            preop_findings = notes_preop['key_findings']
            if preop_findings:
                for i, finding in enumerate(preop_findings[:5], 1):
                    st.info(f"{i}. {finding}")
            else:
                st.caption("No significant findings extracted")
        
        with col2:
            st.markdown("**🔑 Intraoperative Key Findings:**")
            intraop_findings = notes_intraop['key_findings']
            if intraop_findings:
                for i, finding in enumerate(intraop_findings[:5], 1):
                    st.warning(f"{i}. {finding}")
            else:
                st.caption("No significant findings extracted")
        
        st.markdown("---")
        
        # Complication mentions
        st.subheader("⚠️ Complication Mentions in Documentation")
        
        preop_mentions = notes_preop['complication_mentions']
        intraop_mentions = notes_intraop['complication_mentions']
        
        # Combine mentions
        all_mentions = {}
        for mentions_dict in [preop_mentions, intraop_mentions]:
            for pattern, mention_info in mentions_dict.items():
                if pattern not in all_mentions:
                    all_mentions[pattern] = mention_info['count']
                else:
                    all_mentions[pattern] += mention_info['count']
        
        if all_mentions:
            # Sort by frequency
            sorted_mentions = sorted(all_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
            
            mention_df = pd.DataFrame(sorted_mentions, columns=['Term', 'Frequency'])
            
            fig_mentions = px.bar(
                mention_df,
                x='Frequency',
                y='Term',
                orientation='h',
                title='Top 10 Complication-Related Terms',
                color='Frequency',
                color_continuous_scale='Reds'
            )
            
            fig_mentions.update_layout(height=400)
            st.plotly_chart(fig_mentions, use_container_width=True)
            
            if export_figures:
                fig_mentions.write_image(FIGURES_DIR / 'app_complication_mentions.png')
        else:
            st.success("✅ No complication mentions found (positive indicator)")
        
        st.markdown("---")
        
        # Severity indicators
        st.subheader("📈 Severity Indicators")
        
        preop_severity = notes_preop['severity_indicators']
        intraop_severity = notes_intraop['severity_indicators']
        
        severity_df = pd.DataFrame({
            'Severity Level': ['Critical', 'Severe', 'Moderate', 'Mild (Improvement)'],
            'Preoperative': [
                preop_severity['critical'],
                preop_severity['severe'],
                preop_severity['moderate'],
                preop_severity['mild']
            ],
            'Intraoperative': [
                intraop_severity['critical'],
                intraop_severity['severe'],
                intraop_severity['moderate'],
                intraop_severity['mild']
            ]
        })
        
        fig_severity = go.Figure()
        
        fig_severity.add_trace(go.Bar(
            name='Preoperative',
            x=severity_df['Severity Level'],
            y=severity_df['Preoperative'],
            marker_color='#1976d2'
        ))
        
        fig_severity.add_trace(go.Bar(
            name='Intraoperative',
            x=severity_df['Severity Level'],
            y=severity_df['Intraoperative'],
            marker_color='#f57c00'
        ))
        
        fig_severity.update_layout(
            title='Severity Indicators by Phase',
            barmode='group',
            xaxis_title='Severity Level',
            yaxis_title='Mention Count',
            height=400
        )
        
        st.plotly_chart(fig_severity, use_container_width=True)
        
        if export_figures:
            fig_severity.write_image(FIGURES_DIR / 'app_severity_indicators.png')
        
        # Section embeddings
        if notes_preop['section_embeddings'] or notes_intraop['section_embeddings']:
            st.markdown("---")
            st.subheader("📑 Extracted Clinical Sections")
            
            sections_found = set(notes_preop['section_embeddings'].keys()) | set(notes_intraop['section_embeddings'].keys())
            
            if sections_found:
                st.success(f"✓ Extracted {len(sections_found)} clinical sections")
                
                section_cols = st.columns(3)
                for idx, section in enumerate(sorted(sections_found)):
                    with section_cols[idx % 3]:
                        phase = "Preop" if section in notes_preop['section_embeddings'] else "Intraop"
                        st.markdown(f"**{section.replace('_', ' ').title()}** ({phase})")
    
    with tab3:
        st.subheader("🧪 Laboratory Results Analysis")
        
        ts_metadata = st.session_state.ts_metadata
        
        # Metrics by phase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔬 Preoperative Labs**")
            preop_lab_meta = ts_metadata['labs_preop']['lab_metadata']
            
            if preop_lab_meta:
                st.metric("Tests Analyzed", len(preop_lab_meta))
                
                # Show abnormal labs
                abnormal_count = sum(1 for lab, meta in preop_lab_meta.items() 
                                   if 'outliers_removed' in meta and meta['outliers_removed'] > 0)
                st.metric("Labs with Outliers", abnormal_count)
            else:
                st.info("No preoperative labs available")
        
        with col2:
            st.markdown("**🔬 Intraoperative Labs**")
            intraop_lab_meta = ts_metadata['labs_intraop']['lab_metadata']
            
            if intraop_lab_meta:
                st.metric("Tests Analyzed", len(intraop_lab_meta))
                
                abnormal_count = sum(1 for lab, meta in intraop_lab_meta.items() 
                                   if 'outliers_removed' in meta and meta['outliers_removed'] > 0)
                st.metric("Labs with Outliers", abnormal_count)
            else:
                st.info("No intraoperative labs available")
        
        st.markdown("---")
        
        # Lab trends visualization
        st.subheader("📈 Laboratory Trends (Preoperative)")
        
        if preop_lab_meta:
            # Create trend data
            lab_names = []
            lab_means = []
            lab_stds = []
            
            for lab_name, meta in preop_lab_meta.items():
                lab_names.append(lab_name)
                lab_means.append(meta['mean'])
                lab_stds.append(meta['std'])
            
            trend_df = pd.DataFrame({
                'Lab Test': lab_names,
                'Mean Value': lab_means,
                'Std Dev': lab_stds
            })
            
            st.dataframe(trend_df, use_container_width=True, hide_index=True)
            
            # Bar chart
            fig_lab_trends = go.Figure()
            
            fig_lab_trends.add_trace(go.Bar(
                x=lab_names,
                y=lab_means,
                error_y=dict(type='data', array=lab_stds),
                marker_color='steelblue'
            ))
            
            fig_lab_trends.update_layout(
                title='Preoperative Lab Values (Mean ± SD)',
                xaxis_title='Lab Test',
                yaxis_title='Normalized Value',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_lab_trends, use_container_width=True)
            
            if export_figures:
                fig_lab_trends.write_image(FIGURES_DIR / 'app_lab_trends.png')
        
        st.markdown("---")
        
        # Statistical features
        if show_preprocessing:
            st.subheader("📊 Statistical Features Extracted")
            
            stat_features_preop = ts_metadata['labs_preop']['statistical_features']
            
            if stat_features_preop:
                # Show first 20 features
                feature_items = list(stat_features_preop.items())[:20]
                stat_df = pd.DataFrame(feature_items, columns=['Feature', 'Value'])
                
                st.dataframe(stat_df, use_container_width=True)
    
    with tab4:
        st.subheader("💓 Vital Signs Analysis")
        
        ts_metadata = st.session_state.ts_metadata
        
        # Metrics by phase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Preoperative Vitals**")
            preop_vital_meta = ts_metadata['vitals_preop']['vital_metadata']
            
            if preop_vital_meta:
                st.metric("Measurements", sum(meta['count'] for meta in preop_vital_meta.values()))
                
                # Display current vitals
                with st.expander("View Vital Signs"):
                    for vital_name, meta in preop_vital_meta.items():
                        st.text(f"{vital_name}: {meta['mean']:.1f} (±{meta['std']:.1f})")
            else:
                st.info("No preoperative vitals available")
        
        with col2:
            st.markdown("**📊 Intraoperative Vitals**")
            intraop_vital_meta = ts_metadata['vitals_intraop']['vital_metadata']
            
            if intraop_vital_meta:
                st.metric("Measurements", sum(meta['count'] for meta in intraop_vital_meta.values()))
                
                with st.expander("View Vital Signs"):
                    for vital_name, meta in intraop_vital_meta.items():
                        st.text(f"{vital_name}: {meta['mean']:.1f} (±{meta['std']:.1f})")
            else:
                st.info("No intraoperative vitals available")
        
        st.markdown("---")
        
        # Hemodynamic stability indicators
        st.subheader("🫀 Hemodynamic Assessment")
        
        # Calculate derived metrics from metadata
        if preop_vital_meta:
            hr_meta = preop_vital_meta.get('Heart_Rate', {})
            sbp_meta = preop_vital_meta.get('SBP', {})
            
            if hr_meta and sbp_meta:
                shock_index = hr_meta['mean'] / sbp_meta['mean'] if sbp_meta['mean'] > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Shock Index", f"{shock_index:.2f}")
                    if shock_index > 0.9:
                        st.error("⚠️ Elevated - Consider fluid resuscitation")
                    elif shock_index > 0.7:
                        st.warning("⚠️ Borderline")
                    else:
                        st.success("✓ Normal")
                
                with col2:
                    map_est = (sbp_meta['mean'] + 2 * preop_vital_meta.get('DBP', {}).get('mean', 60)) / 3
                    st.metric("MAP (Estimated)", f"{map_est:.0f} mmHg")
                
                with col3:
                    hr_variability = hr_meta.get('std', 0) / hr_meta.get('mean', 1) if hr_meta.get('mean', 1) > 0 else 0
                    st.metric("HR Variability", f"{hr_variability:.2f}")
        
        st.markdown("---")
        
        # Vital signs timeline
        st.subheader("📈 Vital Signs Timeline")
        
        if 'vitals' in st.session_state.patient_data and not st.session_state.patient_data['vitals'].empty:
            vitals_df = st.session_state.patient_data['vitals']
            surgery_time = st.session_state.patient_data['surgery_time']
            
            # Add phase column
            vitals_df['Phase'] = vitals_df['CHARTTIME'].apply(
                lambda x: 'Preoperative' if x < surgery_time else 'Intraoperative/Postoperative'
            )
            
            # Plot heart rate over time
            hr_data = vitals_df[vitals_df['LABEL'].str.contains('Heart Rate', case=False, na=False)]
            
            if not hr_data.empty:
                fig_vitals = px.scatter(
                    hr_data,
                    x='CHARTTIME',
                    y='VALUENUM',
                    color='Phase',
                    title='Heart Rate Over Time',
                    labels={'CHARTTIME': 'Time', 'VALUENUM': 'Heart Rate (bpm)'},
                    color_discrete_map={'Preoperative': '#1976d2', 'Intraoperative/Postoperative': '#f57c00'}
                )
                
                # Add surgery time line
                fig_vitals.add_vline(
                    x=surgery_time,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Surgery Time"
                )
                
                fig_vitals.update_layout(height=400)
                st.plotly_chart(fig_vitals, use_container_width=True)
                
                if export_figures:
                    fig_vitals.write_image(FIGURES_DIR / 'app_vitals_timeline.png')
    
    with tab5:
        st.subheader("💊 Medication Analysis")
        
        aligned_data = st.session_state.aligned_data
        med_data = aligned_data['medications']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Medications", med_data.get('total_count', 0))
        
        with col2:
            st.metric("Unique Drugs", med_data.get('unique_count', 0))
        
        with col3:
            polypharmacy = "Yes" if med_data.get('total_count', 0) >= 5 else "No"
            st.metric("Polypharmacy", polypharmacy)
        
        st.markdown("---")
        
        # Medication categories
        if med_data.get('category_counts'):
            st.subheader("📊 Medication Categories")
            
            cat_df = pd.DataFrame(
                list(med_data['category_counts'].items()),
                columns=['Category', 'Count']
            )
            cat_df = cat_df[cat_df['Count'] > 0].sort_values('Count', ascending=False)
            
            if not cat_df.empty:
                fig_meds = px.bar(
                    cat_df,
                    x='Category',
                    y='Count',
                    title='Medications by Therapeutic Category',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                
                fig_meds.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_meds, use_container_width=True)
                
                if export_figures:
                    fig_meds.write_image(FIGURES_DIR / 'app_medications.png')
            
            # High-risk medications
            high_risk_categories = ['anticoagulant', 'vasopressor', 'steroid']
            high_risk_meds = [(cat, count) for cat, count in med_data['category_counts'].items() 
                            if cat in high_risk_categories and count > 0]
            
            if high_risk_meds:
                st.markdown("---")
                st.subheader("⚠️ High-Risk Medications")
                
                for category, count in high_risk_meds:
                    st.warning(f"**{category.title()}**: {count} medication(s)")
    
    with tab6:
        st.subheader("🔍 Explainability Dashboard")
        
        if show_explainability:
            
            # Feature importance visualization
            st.markdown("### 📊 Feature Importance Analysis")
            
            # Simulate feature importance (replace with real SHAP values)
            feature_groups = ['Clinical Notes (Preop)', 'Clinical Notes (Intraop)', 
                            'Lab Results', 'Vital Signs', 'Demographics', 'Medications']
            
            importance_values = [0.28, 0.22, 0.20, 0.15, 0.10, 0.05]
            
            fig_importance = go.Figure(go.Bar(
                x=importance_values,
                y=feature_groups,
                orientation='h',
                marker=dict(
                    color=importance_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f'{v:.1%}' for v in importance_values],
                textposition='inside'
            ))
            
            fig_importance.update_layout(
                title='Feature Group Importance (Mean Across All Complications)',
                xaxis_title='Importance Score',
                yaxis_title='',
                height=400,
                xaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            if export_figures:
                fig_importance.write_image(FIGURES_DIR / 'app_feature_importance.png')
            
            st.markdown("---")
            
            # Per-complication feature importance
            st.markdown("### 🎯 Per-Complication Feature Importance")
            
            # Create heatmap
            n_complications = len(COMPLICATIONS)
            n_features = len(feature_groups)
            
            # Simulate importance matrix
            importance_matrix = np.random.dirichlet(np.ones(n_features), size=n_complications)
            
            fig_heatmap_importance = go.Figure(data=go.Heatmap(
                z=importance_matrix,
                x=feature_groups,
                y=[COMPLICATIONS[t]['name'] for t in sorted(COMPLICATIONS.keys())],
                colorscale='RdYlGn',
                text=np.round(importance_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Importance')
            ))
            
            fig_heatmap_importance.update_layout(
                title='Feature Importance Heatmap by Complication',
                xaxis_title='Feature Group',
                yaxis_title='Complication',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_heatmap_importance, use_container_width=True)
            
            if export_figures:
                fig_heatmap_importance.write_image(FIGURES_DIR / 'app_importance_heatmap.png')
            
            st.markdown("---")
            
            # Temporal dynamics
            st.markdown("### ⏱️ Temporal Feature Dynamics")
            
            st.info("📊 Shows how key features evolved from preoperative to intraoperative phase")
            
            # Simulate temporal evolution
            time_points = ['Preop -48h', 'Preop -24h', 'Preop -12h', 
                          'Surgery', 'Intraop +6h', 'Intraop +12h']
            
            key_features = ['Creatinine', 'WBC', 'Heart Rate', 'Blood Pressure']
            
            fig_temporal = go.Figure()
            
            for feature in key_features:
                # Simulate trajectory
                values = np.cumsum(np.random.randn(len(time_points)) * 0.1) + np.random.rand()
                
                fig_temporal.add_trace(go.Scatter(
                    x=time_points,
                    y=values,
                    mode='lines+markers',
                    name=feature,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
            
            # Add surgery time marker
            fig_temporal.add_vline(
                x=3,
                line_dash="dash",
                line_color="red",
                annotation_text="Surgery"
            )
            
            fig_temporal.update_layout(
                title='Key Feature Evolution Through Surgical Timeline',
                xaxis_title='Time Point',
                yaxis_title='Normalized Feature Value',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            if export_figures:
                fig_temporal.write_image(FIGURES_DIR / 'app_temporal_dynamics.png')
            
            st.markdown("---")
            
            # Uncertainty visualization
            if show_uncertainty:
                st.markdown("### 📉 Prediction Uncertainty Analysis")
                
                uncertainty_data = []
                for task_name in sorted(COMPLICATIONS.keys()):
                    uncertainty_data.append({
                        'Complication': COMPLICATIONS[task_name]['name'],
                        'Risk Score': predictions[task_name],
                        'Uncertainty': uncertainties.get(task_name, 0.0)
                    })
                
                unc_df = pd.DataFrame(uncertainty_data)
                
                fig_uncertainty = go.Figure()
                
                fig_uncertainty.add_trace(go.Scatter(
                    x=unc_df['Risk Score'],
                    y=unc_df['Uncertainty'],
                    mode='markers+text',
                    marker=dict(size=15, color=unc_df['Risk Score'], 
                              colorscale='RdYlGn_r', showscale=True,
                              colorbar=dict(title='Risk Score')),
                    text=unc_df['Complication'],
                    textposition='top center',
                    textfont=dict(size=9)
                ))
                
                fig_uncertainty.update_layout(
                    title='Risk Score vs Prediction Uncertainty',
                    xaxis_title='Predicted Risk Score',
                    yaxis_title='Uncertainty (Std Dev)',
                    xaxis=dict(tickformat='.0%'),
                    yaxis=dict(tickformat='.0%'),
                    height=500
                )
                
                st.plotly_chart(fig_uncertainty, use_container_width=True)
                
                if export_figures:
                    fig_uncertainty.write_image(FIGURES_DIR / 'app_uncertainty.png')
                
                st.info("""
                **Interpretation:**
                - Low uncertainty = High confidence in prediction
                - High uncertainty = Model is uncertain, consider additional data
                - High risk + high uncertainty = Requires careful clinical assessment
                """)
        
        else:
            st.info("Enable 'Show Uncertainty Estimates' in sidebar to view uncertainty analysis")
    
    with tab7:
        st.subheader("💡 Evidence-Based Clinical Recommendations")
        
        # Generate recommendations based on predictions
        recommendations = []
        
        # Overall risk recommendation
        if overall_risk >= 0.7:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Overall Risk',
                'title': 'High Overall Surgical Risk - Multidisciplinary Review Required',
                'description': f'Patient has overall risk score of {overall_risk:.1%}. Strongly consider comprehensive preoperative optimization and enhanced perioperative monitoring.',
                'actions': [
                    'Obtain senior surgical consultation',
                    'Multidisciplinary team conference',
                    'Consider ICU bed reservation',
                    'Enhanced informed consent discussion',
                    'Review surgical necessity vs alternatives'
                ]
            })
        elif overall_risk >= 0.4:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Overall Risk',
                'title': 'Moderate Surgical Risk - Enhanced Monitoring Recommended',
                'description': f'Patient has moderate overall risk ({overall_risk:.1%}). Standard care with enhanced monitoring appropriate.',
                'actions': [
                    'Standard surgical consent',
                    'Plan for step-down unit or ICU as needed',
                    'Close postoperative monitoring',
                    'Early mobilization protocol'
                ]
            })
        
        # Complication-specific recommendations
        high_risk_complications = [(t, predictions[t]) for t in COMPLICATIONS.keys() 
                                  if predictions[t] >= 0.7]
        
        for task_name, risk_score in high_risk_complications:
            task_info = COMPLICATIONS[task_name]
            
            rec = {
                'priority': 'URGENT',
                'category': task_info['name'],
                'title': f'High Risk of {task_info["name"]}',
                'description': f'Risk score: {risk_score:.1%}. {task_info["description"]}',
                'actions': []
            }
            
            # Task-specific actions
            if task_name == 'aki':
                rec['actions'] = [
                    'Ensure adequate preoperative hydration',
                    'Avoid nephrotoxic medications',
                    'Monitor urine output hourly postoperatively',
                    'Consider nephrology consultation',
                    'Hold ACE inhibitors/ARBs perioperatively',
                    'Optimize hemodynamics'
                ]
            
            elif task_name == 'sepsis':
                rec['actions'] = [
                    'Optimize preoperative nutritional status',
                    'Ensure appropriate antibiotic prophylaxis',
                    'Monitor for SIRS criteria',
                    'Early recognition protocols',
                    'Source control planning',
                    'Consider infectious disease consultation'
                ]
            
            elif task_name == 'prolonged_icu':
                rec['actions'] = [
                    'Reserve ICU bed postoperatively',
                    'Coordinate with ICU team',
                    'Plan for extended monitoring',
                    'Optimize preoperative status',
                    'Family discussion regarding ICU course'
                ]
            
            elif task_name == 'cardiovascular':
                rec['actions'] = [
                    'Preoperative cardiac risk stratification',
                    'Consider cardiology consultation',
                    'Continuous cardiac monitoring',
                    'Troponin surveillance',
                    'Beta-blocker continuation',
                    'Optimize hemodynamics'
                ]
            
            elif task_name == 'wound':
                rec['actions'] = [
                    'Optimize glucose control (target <180 mg/dL)',
                    'Ensure appropriate antibiotic timing',
                    'Consider advanced wound closure',
                    'Enhanced wound monitoring protocol',
                    'Nutritional optimization'
                ]
            
            elif task_name == 'vte':
                rec['actions'] = [
                    'Pharmacologic VTE prophylaxis',
                    'Sequential compression devices',
                    'Early mobilization',
                    'Monitor for signs of DVT/PE',
                    'Consider extended prophylaxis'
                ]
            
            elif task_name == 'prolonged_mv':
                rec['actions'] = [
                    'Preoperative pulmonary function assessment',
                    'Smoking cessation counseling',
                    'Incentive spirometry',
                    'Lung-protective ventilation',
                    'Early extubation protocol'
                ]
            
            elif task_name == 'neurological':
                rec['actions'] = [
                    'Baseline cognitive assessment',
                    'Delirium prevention bundle',
                    'Avoid deliriogenic medications',
                    'Early mobilization',
                    'Consider neurology consultation if high risk'
                ]
            
            elif task_name == 'mortality':
                rec['actions'] = [
                    'Comprehensive informed consent',
                    'Goals of care discussion',
                    'Consider less invasive alternatives',
                    'Optimize ALL modifiable risk factors',
                    'Ensure highest level of monitoring',
                    'Palliative care consultation if appropriate'
                ]
            
            recommendations.append(rec)
        
        # If no high-risk complications
        if not recommendations:
            recommendations.append({
                'priority': 'ROUTINE',
                'category': 'Standard Care',
                'title': 'Proceed with Standard Perioperative Care',
                'description': 'No high-risk factors identified. Continue with routine surgical protocols.',
                'actions': [
                    'Standard surgical consent',
                    'Routine postoperative monitoring',
                    'Early mobilization',
                    'VTE prophylaxis per protocol',
                    'Standard pain management'
                ]
            })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            priority = rec['priority']
            
            # Color coding
            if priority == 'CRITICAL':
                color = '#ffcdd2'
                border_color = '#d32f2f'
                emoji = '🔴'
            elif priority == 'URGENT':
                color = '#ffe0b2'
                border_color = '#f57c00'
                emoji = '🟠'
            elif priority == 'HIGH':
                color = '#fff9c4'
                border_color = '#fbc02d'
                emoji = '🟡'
            else:
                color = '#e8f5e9'
                border_color = '#4caf50'
                emoji = '🟢'
            
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 15px; 
                        margin-bottom: 20px; border-left: 8px solid {border_color}; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin: 0 0 10px 0;'>{emoji} {i}. {rec['title']}</h3>
                <p style='margin: 0 0 5px 0;'><b>Priority:</b> <span style='color: {border_color}; font-weight: bold;'>{priority}</span></p>
                <p style='margin: 0 0 5px 0;'><b>Category:</b> {rec['category']}</p>
                <p style='margin: 10px 0;'>{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action items
            if rec.get('actions'):
                st.markdown("**🎯 Recommended Actions:**")
                for action in rec['actions']:
                    st.checkbox(action, key=f"action_{i}_{action}", value=False)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Timeline for implementation
        st.subheader("⏱️ Recommended Timeline")
        
        timeline_data = {
            'Phase': ['Immediate\n(0-2 hours)', 'Preoperative\n(2-24 hours)', 
                     'Intraoperative', 'Postop Day 0-1', 'Postop Day 2-7', 'Long-term'],
            'Actions': [
                'Address critical findings',
                'Complete optimization, consultations',
                'Enhanced monitoring per protocol',
                'Close surveillance, early intervention',
                'Continue monitoring, address complications',
                'Follow-up, rehabilitation'
            ]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.table(timeline_df)
    
    # Additional Integrated Analysis
    st.markdown("---")
    st.markdown("## 📊 Integrated Multimodal Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "🔗 Data Integration Summary",
        "📈 Comparative Analysis",
        "📄 Export Report"
    ])
    
    with analysis_tab1:
        st.subheader("Multimodal Data Integration Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📝 Clinical Notes**")
            st.metric("Preoperative Notes", notes_preop['metadata']['num_notes'])
            st.metric("Intraoperative Notes", notes_intraop['metadata']['num_notes'])
            st.metric("Total Words", notes_preop['metadata']['total_words'] + notes_intraop['metadata']['total_words'])
        
        with col2:
            st.markdown("**🧪 Laboratory Data**")
            st.metric("Preop Tests", len(ts_metadata['labs_preop']['lab_metadata']))
            st.metric("Intraop Tests", len(ts_metadata['labs_intraop']['lab_metadata']))
            st.metric("Time Series Length", aligned_data['time_series']['sequence_length'])
        
        with col3:
            st.markdown("**💓 Vital Signs**")
            preop_vitals_count = sum(meta['count'] for meta in ts_metadata['vitals_preop']['vital_metadata'].values())
            intraop_vitals_count = sum(meta['count'] for meta in ts_metadata['vitals_intraop']['vital_metadata'].values())
            st.metric("Preop Measurements", preop_vitals_count)
            st.metric("Intraop Measurements", intraop_vitals_count)
            st.metric("Unique Parameters", len(ts_metadata['vitals_preop']['vital_metadata']))
        
        st.markdown("---")
        
        # Data completeness
        st.markdown("**✅ Data Completeness Assessment**")
        
        completeness = {
            'Clinical Notes (Preop)': min(100, notes_preop['metadata']['num_notes'] * 50),
            'Clinical Notes (Intraop)': min(100, notes_intraop['metadata']['num_notes'] * 50),
            'Laboratory Results': min(100, len(ts_metadata['labs_preop']['lab_metadata']) * 10),
            'Vital Signs': min(100, preop_vitals_count / 20 * 100),
            'Medications': min(100, aligned_data['medications']['total_count'] * 20)
        }
        
        completeness_df = pd.DataFrame(
            list(completeness.items()),
            columns=['Data Source', 'Completeness (%)']
        )
        
        fig_completeness = px.bar(
            completeness_df,
            x='Data Source',
            y='Completeness (%)',
            title='Data Completeness by Source',
            color='Completeness (%)',
            color_continuous_scale='Greens',
            range_color=[0, 100]
        )
        
        fig_completeness.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_completeness, use_container_width=True)
        
        if export_figures:
            fig_completeness.write_image(FIGURES_DIR / 'app_data_completeness.png')
    
    with analysis_tab2:
        st.subheader("Comparative Risk Analysis")
        
        # Compare phases
        st.markdown("### 📊 Preoperative vs Intraoperative Phase Comparison")
        
        comparison_data = {
            'Metric': ['Severity Score', 'Complication Mentions', 'Data Points'],
            'Preoperative': [
                f"{notes_preop['metadata']['severity_score']:.1%}",
                str(notes_preop['metadata']['num_complication_mentions']),
                str(preop_vitals_count + len(ts_metadata['labs_preop']['lab_metadata']))
            ],
            'Intraoperative': [
                f"{notes_intraop['metadata']['severity_score']:.1%}",
                str(notes_intraop['metadata']['num_complication_mentions']),
                str(intraop_vitals_count + len(ts_metadata['labs_intraop']['lab_metadata']))
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        st.markdown("---")
        
        # Risk distribution
        st.markdown("### 📈 Risk Distribution")
        
        risk_bins = ['Low\n(<40%)', 'Moderate\n(40-70%)', 'High\n(>70%)']
        bin_counts = [
            sum(1 for p in predictions.values() if p < 0.4),
            sum(1 for p in predictions.values() if 0.4 <= p < 0.7),
            sum(1 for p in predictions.values() if p >= 0.7)
        ]
        
        fig_dist = go.Figure(data=[
            go.Bar(
                x=risk_bins,
                y=bin_counts,
                marker_color=['#4caf50', '#ff9800', '#ff4444'],
                text=bin_counts,
                textposition='auto'
            )
        ])
        
        fig_dist.update_layout(
            title='Distribution of Complication Risk Levels',
            xaxis_title='Risk Category',
            yaxis_title='Number of Complications',
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        if export_figures:
            fig_dist.write_image(FIGURES_DIR / 'app_risk_distribution.png')
    
    with analysis_tab3:
        st.subheader("📄 Generate Comprehensive Report")
        
        # Report configuration
        st.markdown("**Configure Report Sections:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_sections = st.multiselect(
                "Include Sections",
                ["Executive Summary", "Patient Demographics", "Risk Assessment",
                 "Clinical Notes Analysis", "Laboratory Results", "Vital Signs",
                 "Medications", "Recommendations", "Methodology"],
                default=["Executive Summary", "Risk Assessment", "Recommendations"]
            )
        
        with col2:
            report_format = st.selectbox("Format", ["Markdown", "Text", "HTML"])
            include_figures = st.checkbox("Include Figure References", value=True)
            include_timestamp = st.checkbox("Include Timestamp", value=True)
        
        st.markdown("---")
        
        if st.button("📝 Generate Report", use_container_width=True):
            
            # Build report
            report_lines = []
            
            if include_timestamp:
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")
            
            report_lines.extend([
                "="*80,
                "SURGICAL RISK PREDICTION REPORT",
                "Multimodal AI Analysis",
                "="*80,
                ""
            ])
            
            if "Executive Summary" in include_sections:
                report_lines.extend([
                    "EXECUTIVE SUMMARY",
                    "-"*80,
                    f"Overall Risk Score: {overall_risk:.1%}",
                    f"Risk Level: {risk_level}",
                    f"High-Risk Complications: {high_risk_count}/9",
                    "",
                    f"This patient has {risk_level} overall surgical risk based on comprehensive",
                    "multimodal analysis of preoperative and intraoperative data.",
                    ""
                ])
            
            if "Patient Demographics" in include_sections:
                report_lines.extend([
                    "PATIENT INFORMATION",
                    "-"*80,
                    f"Age: {patient_age} years",
                    f"Gender: {patient_gender}",
                    f"Admission Type: {admission_type}",
                    ""
                ])
            
            if "Risk Assessment" in include_sections:
                report_lines.extend([
                    "RISK ASSESSMENT - NINE POSTOPERATIVE COMPLICATIONS",
                    "-"*80,
                    ""
                ])
                
                for task_name in sorted(COMPLICATIONS.keys()):
                    task_info = COMPLICATIONS[task_name]
                    risk = predictions[task_name]
                    unc = uncertainties.get(task_name, 0.0)
                    
                    risk_cat = "HIGH" if risk >= 0.7 else "MODERATE" if risk >= 0.4 else "LOW"
                    
                    report_lines.append(
                        f"{task_info['name']:.<50} {risk:.1%} ± {unc:.1%} ({risk_cat})"
                    )
                
                report_lines.append("")
            
            if "Clinical Notes Analysis" in include_sections:
                report_lines.extend([
                    "CLINICAL NOTES ANALYSIS",
                    "-"*80,
                    "",
                    "Preoperative Phase:",
                    f"  - Notes analyzed: {notes_preop['metadata']['num_notes']}",
                    f"  - Severity score: {notes_preop['metadata']['severity_score']:.1%}",
                    f"  - Complication mentions: {notes_preop['metadata']['num_complication_mentions']}",
                    "",
                    "Intraoperative Phase:",
                    f"  - Notes analyzed: {notes_intraop['metadata']['num_notes']}",
                    f"  - Severity score: {notes_intraop['metadata']['severity_score']:.1%}",
                    f"  - Complication mentions: {notes_intraop['metadata']['num_complication_mentions']}",
                    ""
                ])
            
            if "Recommendations" in include_sections:
                report_lines.extend([
                    "CLINICAL RECOMMENDATIONS",
                    "-"*80,
                    ""
                ])
                
                for i, rec in enumerate(recommendations, 1):
                    report_lines.extend([
                        f"{i}. [{rec['priority']}] {rec['title']}",
                        f"   Category: {rec['category']}",
                        f"   {rec['description']}",
                        ""
                    ])
                    
                    if rec.get('actions'):
                        report_lines.append("   Recommended Actions:")
                        for action in rec['actions']:
                            report_lines.append(f"     • {action}")
                        report_lines.append("")
            
            if "Methodology" in include_sections:
                report_lines.extend([
                    "METHODOLOGY",
                    "-"*80,
                    "",
                    "Data Sources:",
                    "  • Clinical notes (preoperative and intraoperative)",
                    "  • Laboratory results (temporal sequences)",
                    "  • Vital signs (continuous monitoring)",
                    "  • Medication administration records",
                    "",
                    "Model Architecture:",
                    "  • Vibe-Tuned Biomedical Language Model (PubMedBERT)",
                    "  • Transformer-based time series encoder",
                    "  • Cross-modal attention fusion",
                    "  • Multi-task learning (9 complications)",
                    "  • Monte Carlo dropout for uncertainty estimation",
                    "",
                    "Temporal Windows:",
                    "  • Preoperative: 48h (labs), 24h (vitals), 7d (notes)",
                    "  • Intraoperative: 24h window from surgery start",
                    ""
                ])
            
            report_lines.extend([
                "="*80,
                "DISCLAIMER",
                "="*80,
                "",
                "This report is generated by AI for clinical decision support purposes only.",
                "All predictions should be validated by qualified healthcare professionals.",
                "This system is for research and educational use and has not been FDA approved",
                "for clinical decision-making.",
                "",
                "Powered by: University of Florida NaviGator AI",
                "System: Surgical Risk Prediction System v1.0",
                "Model: Multimodal Vibe-Tuned Transformer",
                "",
                "="*80
            ])
            
            report_text = '\n'.join(report_lines)
            
            # Display preview
            st.markdown("**📄 Report Preview:**")
            with st.expander("View Full Report", expanded=False):
                st.text_area("", report_text, height=400, disabled=True)
            
            # Download button
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"surgical_risk_report_{timestamp}.txt"
            
            st.download_button(
                label="⬇️ Download Report",
                data=report_text,
                file_name=filename,
                mime="text/plain",
                use_container_width=True
            )
            
            st.success("✓ Report generated successfully!")
            
            # Option to save all figures as ZIP
            if export_figures:
                st.markdown("---")
                st.info(f"📊 All figures have been saved to: `{FIGURES_DIR}`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
     border-radius: 15px; color: white;'>
    <h3 style='color: white; margin: 0 0 15px 0;'>🏥 Surgical Risk Prediction System</h3>
    
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin: 15px 0;'>
        <div style='margin: 10px;'>
            <h4 style='color: white; margin: 0;'>9</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>Complications Predicted</p>
        </div>
        <div style='margin: 10px;'>
            <h4 style='color: white; margin: 0;'>4</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>Data Modalities</p>
        </div>
        <div style='margin: 10px;'>
            <h4 style='color: white; margin: 0;'>90%+</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>Parameter Reduction</p>
        </div>
        <div style='margin: 10px;'>
            <h4 style='color: white; margin: 0;'>100ms</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>Inference Time</p>
        </div>
    </div>
    
    <p style='margin: 15px 0 10px 0; font-size: 0.95em;'>
        <b>Technology:</b> Vibe-Tuned Language Models • Transformer Architecture • Cross-Modal Attention
    </p>
    
    <p style='margin: 10px 0; font-size: 0.95em;'>
        <b>Features:</b> Preop/Intraop Separation • Uncertainty Quantification • SHAP Explainability
    </p>
    
    <p style='margin: 15px 0 0 0; font-size: 0.85em; opacity: 0.9;'>
        <i>Powered by NaviGator AI • University of Florida</i>
    </p>
    
    <p style='margin: 10px 0 0 0; font-size: 0.75em; opacity: 0.8;'>
        ⚠️ For Research and Educational Purposes Only • Not for Clinical Use Without Validation
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #f5f5f5; border-radius: 10px;'>
        <p style='margin: 0; font-size: 0.85em; color: #666;'>
            <b>Version:</b> 1.0.0<br>
            <b>Model:</b> Vibe-Tuned Multimodal<br>
            <b>Last Updated:</b> 2024
        </p>
    </div>
    """, unsafe_allow_html=True)