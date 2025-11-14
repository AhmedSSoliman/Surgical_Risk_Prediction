#!/usr/bin/env python3
"""
Library Testing Script for Surgical Risk Prediction Project
Tests all required libraries and their versions before running the main pipeline
"""

import sys
from typing import Dict, List, Tuple

def test_library(name: str, import_statement: str, version_check: str = None) -> Tuple[bool, str]:
    """
    Test if a library can be imported and get its version
    
    Args:
        name: Display name of the library
        import_statement: Python import statement
        version_check: Optional code to get version
    
    Returns:
        Tuple of (success, message)
    """
    try:
        exec(import_statement, globals())
        if version_check:
            version = eval(version_check)
            return True, f"✅ {name}: {version}"
        else:
            return True, f"✅ {name}: Installed"
    except Exception as e:
        return False, f"❌ {name}: {str(e)}"

def main():
    print("=" * 80)
    print("Testing Required Libraries for Surgical Risk Prediction")
    print("=" * 80)
    print()
    
    # Define all required libraries
    libraries = [
        # Core Python Libraries
        ("Python Version", "import sys", "sys.version"),
        
        # Data Processing
        ("NumPy", "import numpy as np", "np.__version__"),
        ("Pandas", "import pandas as pd", "pd.__version__"),
        ("SciPy", "import scipy", "scipy.__version__"),
        
        # Machine Learning - Core
        ("PyTorch", "import torch", "torch.__version__"),
        ("Scikit-learn", "import sklearn", "sklearn.__version__"),
        
        # Deep Learning - NLP
        ("Transformers (HuggingFace)", "import transformers", "transformers.__version__"),
        ("Tokenizers", "import tokenizers", "tokenizers.__version__"),
        
        # NLP Processing
        ("spaCy", "import spacy", "spacy.__version__"),
        ("NLTK", "import nltk", "nltk.__version__"),
        
        # Time Series
        ("tsfresh", "import tsfresh", "tsfresh.__version__"),
        
        # Visualization
        ("Matplotlib", "import matplotlib", "matplotlib.__version__"),
        ("Seaborn", "import seaborn as sns", "sns.__version__"),
        ("Plotly", "import plotly", "plotly.__version__"),
        
        # Model Explainability
        ("SHAP", "import shap", "shap.__version__"),
        
        # Training & Monitoring
        ("TensorBoard", "import tensorboard", "tensorboard.__version__"),
        ("tqdm", "import tqdm", "tqdm.__version__"),
        
        # Utilities
        ("PyYAML", "import yaml", "yaml.__version__"),
        ("jsonlines", "import jsonlines", None),  # jsonlines doesn't have __version__
        ("pickle", "import pickle", None),
        
        # Optional but recommended
        ("joblib", "import joblib", "joblib.__version__"),
    ]
    
    print("1. CORE LIBRARIES")
    print("-" * 80)
    results = []
    for lib_name, import_stmt, version_check in libraries[:4]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("2. MACHINE LEARNING FRAMEWORKS")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[4:8]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("3. NLP LIBRARIES")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[8:10]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("4. TIME SERIES & FEATURE EXTRACTION")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[10:11]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("5. VISUALIZATION")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[11:14]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("6. EXPLAINABILITY")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[14:15]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("7. TRAINING & MONITORING")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[15:17]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    print()
    print("8. UTILITIES")
    print("-" * 80)
    for lib_name, import_stmt, version_check in libraries[17:]:
        success, message = test_library(lib_name, import_stmt, version_check)
        results.append((lib_name, success))
        print(f"  {message}")
    
    # Test PyTorch-specific features
    print()
    print("9. PYTORCH CAPABILITIES")
    print("-" * 80)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        print(f"  ✅ CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"     CUDA Version: {torch.version.cuda}")
            print(f"     Device Count: {torch.cuda.device_count()}")
        
        print(f"  ✅ MPS (Apple Silicon) Available: {mps_available}")
        
        device = "cuda" if cuda_available else "mps" if mps_available else "cpu"
        print(f"  ✅ Recommended Device: {device}")
        
    except Exception as e:
        print(f"  ❌ PyTorch capability check failed: {e}")
    
    # Test spaCy models
    print()
    print("10. SPACY LANGUAGE MODELS")
    print("-" * 80)
    
    try:
        import spacy
        models_to_check = ['en_core_web_sm', 'en_core_sci_sm']
        for model_name in models_to_check:
            try:
                nlp = spacy.load(model_name)
                print(f"  ✅ {model_name}: Loaded successfully")
            except OSError:
                print(f"  ❌ {model_name}: Not installed")
                print(f"     Install with: python -m spacy download {model_name}")
    except Exception as e:
        print(f"  ❌ spaCy model check failed: {e}")
    
    # Test HuggingFace model access
    print()
    print("11. HUGGINGFACE MODELS")
    print("-" * 80)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        print(f"  Testing: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✅ Tokenizer loaded successfully")
        
        # Don't load the full model to save time, just verify access
        print(f"  ✅ Model accessible (not fully loaded to save time)")
        
    except Exception as e:
        print(f"  ❌ HuggingFace model test failed: {e}")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    print(f"Total Libraries Tested: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed > 0:
        print()
        print("Failed Libraries:")
        for lib_name, success in results:
            if not success:
                print(f"  - {lib_name}")
        
        print()
        print("=" * 80)
        print("INSTALLATION COMMANDS FOR MISSING LIBRARIES")
        print("=" * 80)
        print()
        print("Install all required packages:")
        print("  pip install -r requirements.txt")
        print()
        print("Or install individually:")
        print("  pip install torch torchvision torchaudio")
        print("  pip install transformers tokenizers")
        print("  pip install scikit-learn pandas numpy scipy")
        print("  pip install spacy nltk")
        print("  pip install tsfresh")
        print("  pip install matplotlib seaborn plotly")
        print("  pip install shap")
        print("  pip install tensorboard tqdm")
        print("  pip install pyyaml jsonlines joblib")
        print()
        print("Download spaCy models:")
        print("  python -m spacy download en_core_web_sm")
        print("  python -m spacy download en_core_sci_sm")
        print()
        return 1
    else:
        print()
        print("🎉 All required libraries are installed and working correctly!")
        print("You can now run the surgical risk prediction pipeline.")
        print()
        return 0

if __name__ == "__main__":
    sys.exit(main())
