surgical_risk_prediction/
├── config.py
├── requirements.txt
├── README.md
├── run_pipeline.py
├── preprocessing/
│   ├── __init__.py
│   ├── preprocess_time_series.py
│   ├── preprocess_notes.py
│   └── align_modalities.py
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── model.py
│   └── vibe_tuning.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py
│   ├── attention_viz.py
│   └── feature_importance.py
├── visualization/
│   ├── __init__.py
│   ├── plot_results.py
│   └── plot_explainability.py
├── utils/
│   ├── __init__.py
│   └── utils.py
└── app.py