# config.py
"""
Complete Configuration for Surgical Risk Prediction System
"""

import os
from pathlib import Path
from datetime import timedelta

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
MODEL_DIR = BASE_DIR / 'models' / 'checkpoints'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'
LOGS_DIR = BASE_DIR / 'logs'
MIMIC_PATH = os.getenv('MIMIC_PATH', './mimic-iii-clinical-database-1.4')

# Create directories
for dir_path in [DATA_DIR, RAW_DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Nine Postoperative Complications
COMPLICATIONS = {
    'prolonged_icu': {
        'name': 'Prolonged ICU Stay',
        'description': 'ICU stay > 48 hours',
        'threshold': 48,
        'weight': 0.8,
        'type': 'binary'
    },
    'aki': {
        'name': 'Acute Kidney Injury',
        'description': 'Postoperative kidney dysfunction',
        'icd9_codes': ['584', '586', '997.5'],
        'weight': 1.0,
        'type': 'binary'
    },
    'prolonged_mv': {
        'name': 'Prolonged Mechanical Ventilation',
        'description': 'Mechanical ventilation > 48 hours',
        'threshold': 48,
        'weight': 1.0,
        'type': 'binary'
    },
    'wound': {
        'name': 'Wound Complications',
        'description': 'Surgical site infections',
        'icd9_codes': ['998.3', '998.5', '998.6'],
        'weight': 0.6,
        'type': 'binary'
    },
    'neurological': {
        'name': 'Neurological Complications',
        'description': 'Stroke, delirium, or CNS complications',
        'icd9_codes': ['430', '431', '432', '433', '434', '435', '436', '437', '438', '293', '348'],
        'weight': 1.0,
        'type': 'binary'
    },
    'sepsis': {
        'name': 'Sepsis',
        'description': 'Systemic infection response',
        'icd9_codes': ['038', '995.91', '995.92', '785.52'],
        'weight': 1.0,
        'type': 'binary'
    },
    'cardiovascular': {
        'name': 'Cardiovascular Complications',
        'description': 'MI, arrhythmia, cardiac arrest',
        'icd9_codes': ['410', '411', '427', '997.1', '785.51'],
        'weight': 1.0,
        'type': 'binary'
    },
    'vte': {
        'name': 'Venous Thromboembolism',
        'description': 'DVT or pulmonary embolism',
        'icd9_codes': ['415.1', '451', '453'],
        'weight': 0.8,
        'type': 'binary'
    },
    'mortality': {
        'name': 'In-Hospital Mortality',
        'description': 'Death during hospitalization',
        'weight': 1.0,
        'type': 'binary'
    }
}

# Temporal Windows Configuration
TEMPORAL_WINDOWS = {
    'preoperative': {
        'labs': timedelta(hours=48),  # 48 hours before surgery
        'vitals': timedelta(hours=24),  # 24 hours before surgery
        'notes': timedelta(days=7),  # 7 days before surgery
        'medications': timedelta(days=1)  # 1 day before surgery
    },
    'intraoperative': {
        'duration': timedelta(hours=24),  # Consider 24 hour window as intraop
        'vitals': timedelta(hours=12),  # Intraop vitals
        'notes': ['OR', 'Anesthesia', 'Operative', 'Surgical']  # Intraop note keywords
    },
    'postoperative': {
        'labs': timedelta(hours=72),  # 72 hours after surgery
        'vitals': timedelta(hours=48),  # 48 hours after surgery
        'notes': timedelta(days=7),  # 7 days after surgery
        'outcome_window': timedelta(days=30)  # 30-day outcomes
    }
}

# Time Series Configuration
TIME_SERIES_CONFIG = {
    'lab_window_hours': 48,
    'vital_window_hours': 24,
    'sampling_rate': '1H',  # 1 hour
    'max_sequence_length': 72,
    'imputation_method': 'forward_fill',
    'normalization': 'robust',
    'outlier_threshold': 3.0
}

# Lab Items (MIMIC-III ITEMIDs)
LAB_ITEMS = {
    'Hemoglobin': [50811, 51222],
    'Hematocrit': [50810, 51221],
    'WBC': [51300, 51301],
    'Platelet': [51265],
    'Sodium': [50824, 50983],
    'Potassium': [50822, 50971],
    'Chloride': [50806, 50902],
    'Bicarbonate': [50803, 50882],
    'BUN': [51006],
    'Creatinine': [50912],
    'Glucose': [50809, 50931],
    'Calcium': [50893],
    'Magnesium': [50960],
    'Phosphate': [50970],
    'Lactate': [50813],
    'INR': [51237],
    'PTT': [51275],
    'Albumin': [50862],
    'Bilirubin': [50885],
    'ALT': [50861],
    'AST': [50878]
}

# Vital Items (MIMIC-III ITEMIDs)
VITAL_ITEMS = {
    'Heart_Rate': [211, 220045],
    'SBP': [51, 442, 455, 6701, 220179, 220050],
    'DBP': [8368, 8440, 8441, 8555, 220180, 220051],
    'MBP': [456, 52, 6702, 443, 220052, 220181, 225312],
    'Respiratory_Rate': [618, 615, 220210, 224690],
    'Temperature': [223761, 678],
    'SpO2': [646, 220277],
    'FiO2': [190, 3420, 3422, 223835]
}

# Clinical Notes Categories with Temporal Classification
NOTE_CATEGORIES = {
    'preoperative': [
        'Discharge summary',
        'History and physical',
        'Consult',
        'Case Management'
    ],
    'intraoperative': [
        'OR',
        'Anesthesia',
        'Operative',
        'Procedure'
    ],
    'postoperative': [
        'Physician',
        'Nursing',
        'Progress',
        'Radiology',
        'ECG',
        'Echo'
    ]
}

# Model Configuration
MODEL_CONFIG = {
    # Vibe-Tuning parameters for Small Language Model
    'vibe_tuning': {
        'enabled': True,
        'base_model': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'model_type': 'bert',  # 'bert', 'roberta', 'biogpt'
        'hidden_size': 768,
        'max_length': 512,
        'frozen_layers': 6,  # Freeze first 6 layers
        'adapter_size': 64,
        'adapter_type': 'lora',  # 'lora', 'prefix', 'adapter'
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01
    },
    
    # Time series encoder
    'time_series': {
        'model_type': 'transformer',  # 'transformer', 'lstm', 'gru', 'tcn'
        'input_size': None,  # Will be set dynamically
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.2,
        'bidirectional': True,
        'use_positional_encoding': True
    },
    
    # Multimodal fusion
    'fusion': {
        'fusion_type': 'cross_attention',  # 'cross_attention', 'concat', 'gated', 'bilinear'
        'hidden_sizes': [512, 256],
        'num_attention_heads': 8,
        'attention_dropout': 0.2,
        'projection_dim': 256,
        'use_layer_norm': True
    },
    
    # Multi-task prediction heads
    'prediction_heads': {
        'shared_layers': [256, 128],
        'task_specific_layers': [64, 32],
        'dropout': 0.3,
        'activation': 'relu',
        'use_batch_norm': True,
        'uncertainty_estimation': True,
        'monte_carlo_dropout': True,
        'mc_samples': 10
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6,
    
    'validation_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    
    # Loss function
    'loss_function': 'focal_loss',  # 'focal_loss', 'weighted_bce', 'asl'
    'focal_loss_gamma': 2.0,
    'focal_loss_alpha': 0.25,
    
    # Class imbalance handling
    'use_class_weights': True,
    'oversampling': False,
    'undersampling': False,
    'smote': False,
    
    # Optimizer
    'optimizer': 'adamw',  # 'adamw', 'adam', 'sgd'
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    
    # Learning rate scheduler
    'scheduler': 'plateau',  # 'cosine_with_warmup', 'plateau', 'step' (cosine_with_warmup causes segfaults on CPU)
    'warmup_epochs': 5,
    
    # Mixed precision training (disabled for CPU)
    'use_mixed_precision': False,  # Only works with CUDA
    'gradient_accumulation_steps': 1,
    
    # Regularization
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 0.0,
    
    # Logging and checkpointing
    'log_interval': 10,
    'eval_interval': 100,
    'checkpoint_interval': 5,
    'save_best_only': True,
    'use_wandb': False,
    'use_tensorboard': True,
    
    # Multi-GPU
    'num_gpus': 1,
    'distributed': False
}

# Explainability Configuration
EXPLAINABILITY_CONFIG = {
    'shap': {
        'enabled': True,
        'num_background_samples': 100,
        'explainer_type': 'deep',  # 'deep', 'gradient', 'kernel'
        'batch_size': 32,
        'save_plots': True
    },
    'attention': {
        'enabled': True,
        'visualize_layers': [6, 9, 11],
        'head_reduction': 'mean',  # 'mean', 'max', 'all'
        'save_heatmaps': True
    },
    'lime': {
        'enabled': True,
        'num_samples': 1000,
        'num_features': 20
    },
    'integrated_gradients': {
        'enabled': True,
        'n_steps': 50,
        'internal_batch_size': 8
    },
    'grad_cam': {
        'enabled': True,
        'target_layers': ['fusion_layer', 'final_layer']
    },
    'feature_importance': {
        'enabled': True,
        'method': 'permutation',  # 'permutation', 'ablation'
        'num_repeats': 10
    }
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_format': 'png',  # 'png', 'pdf', 'svg'
    'dpi': 300,
    'figure_size': (12, 8),
    'style': 'darkgrid',
    'color_palette': 'husl',
    'save_figures': True,
    'show_figures': False,
    
    'plots': {
        'training_curves': True,
        'roc_curves': True,
        'pr_curves': True,
        'confusion_matrices': True,
        'calibration_plots': True,
        'attention_heatmaps': True,
        'shap_summary': True,
        'shap_waterfall': True,
        'shap_dependence': True,
        'feature_importance': True,
        'time_series_plots': True,
        'risk_stratification': True,
        'temporal_analysis': True
    }
}

# Evaluation Metrics
EVALUATION_METRICS = [
    'auroc',
    'auprc',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'specificity',
    'sensitivity',
    'npv',
    'ppv',
    'brier_score',
    'ece',  # Expected Calibration Error
    'mce'   # Maximum Calibration Error
]