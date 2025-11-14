"""
Vibe-Tuning Configuration for Surgical Risk Prediction
Optimized for MacBook Pro (Apple Silicon M1/M2/M3)

This script configures Teacher-Student model pairs for distillation via distil labs platform.
"""

import torch
from models.vibe_tuning import VibeTunedBiomedicalEncoder

# ==============================================================================
# MACBOOK PRO OPTIMIZATION
# ==============================================================================

def get_device():
    """
    Get optimal device with priority: CUDA > MPS > CPU
    
    Returns:
        - 'cuda' for NVIDIA GPU (highest priority)
        - 'mps' for Apple Silicon (M1/M2/M3) GPU acceleration
        - 'cpu' as fallback
    """
    if torch.cuda.is_available():
        print("✅ CUDA GPU detected - Using NVIDIA GPU acceleration")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) detected - Using GPU acceleration")
        return torch.device('mps')
    else:
        print("⚠️  No GPU available - Using CPU")
        return torch.device('cpu')

# ==============================================================================
# RECOMMENDED TEACHER-STUDENT PAIRS
# ==============================================================================

VIBE_TUNING_CONFIGS = {
    # Option 1: Best for Medical Reasoning (Recommended)
    'medical_reasoning': {
        'teacher': 'deepseek.r1',
        'student': 'Llama-3.2-3B',
        'description': 'Advanced medical reasoning with good performance',
        'use_case': 'Complex surgical risk analysis, multi-factor predictions',
        'memory_req': '~8-12 GB RAM',
        'training_time': '~2-4 hours on MacBook Pro'
    },
    
    # Option 2: Most Efficient (Fastest)
    'efficient': {
        'teacher': 'Claude 3.5 Sonnet',
        'student': 'SmolLM2-1.7B',
        'description': 'Excellent clinical text understanding, lightweight student',
        'use_case': 'Real-time inference, edge deployment',
        'memory_req': '~6-8 GB RAM',
        'training_time': '~1-2 hours on MacBook Pro'
    },
    
    # Option 3: Production Ready (Open Source)
    'production': {
        'teacher': 'Llama-3.1-405B',
        'student': 'Llama-3.2-1B',
        'description': 'Fully open source, smallest student model',
        'use_case': 'Production deployment, mobile/web apps',
        'memory_req': '~4-6 GB RAM',
        'training_time': '~1-1.5 hours on MacBook Pro'
    },
    
    # Option 4: Balanced (Good all-around)
    'balanced': {
        'teacher': 'GPT-4',
        'student': 'Qwen3-4B',
        'description': 'Strong reasoning with moderate model size',
        'use_case': 'General surgical risk prediction',
        'memory_req': '~8-10 GB RAM',
        'training_time': '~2-3 hours on MacBook Pro'
    },
    
    # Option 5: Ultra-Light (Minimal resources)
    'ultra_light': {
        'teacher': 'Gemini 2 Flash',
        'student': 'SmolLM2-135M',
        'description': 'Fastest training, smallest memory footprint',
        'use_case': 'Prototyping, testing, resource-constrained scenarios',
        'memory_req': '~3-4 GB RAM',
        'training_time': '~30-60 minutes on MacBook Pro'
    }
}

# ==============================================================================
# VIBE-TUNING SETUP FUNCTION
# ==============================================================================

def setup_vibe_tuning(config_name='medical_reasoning', 
                      base_model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                      use_lora=True,
                      use_adapters=True,
                      use_prefix=False):
    """
    Setup Vibe-tuned model for surgical risk prediction on MacBook Pro
    
    Args:
        config_name: One of ['medical_reasoning', 'efficient', 'production', 
                            'balanced', 'ultra_light']
        base_model: Base biomedical encoder (Student model from HuggingFace)
        use_lora: Enable LoRA adapters (recommended: True)
        use_adapters: Enable Adapter layers (recommended: True)
        use_prefix: Enable Prefix tuning (optional: False, uses more memory)
    
    Returns:
        model: VibeTunedBiomedicalEncoder ready for training
        device: torch.device (mps or cpu)
        config: Configuration dictionary
    """
    
    # Get configuration
    if config_name not in VIBE_TUNING_CONFIGS:
        raise ValueError(f"Config '{config_name}' not found. Choose from: {list(VIBE_TUNING_CONFIGS.keys())}")
    
    config = VIBE_TUNING_CONFIGS[config_name]
    
    # Print configuration
    print("\n" + "="*80)
    print(f"🎯 Vibe-Tuning Configuration: {config_name.upper()}")
    print("="*80)
    print(f"  Teacher Model:     {config['teacher']}")
    print(f"  Student Model:     {config['student']}")
    print(f"  Base Encoder:      {base_model}")
    print(f"  Fine-tuning:       LoRA={use_lora}, Adapters={use_adapters}, Prefix={use_prefix}")
    print(f"  Description:       {config['description']}")
    print(f"  Use Case:          {config['use_case']}")
    print(f"  Memory Required:   {config['memory_req']}")
    print(f"  Est. Training:     {config['training_time']}")
    print("="*80 + "\n")
    
    # Get device
    device = get_device()
    
    # Create model config
    model_config = {
        'base_model': base_model,
        'use_lora': use_lora,
        'use_adapters': use_adapters,
        'use_prefix_tuning': use_prefix,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'adapter_hidden_size': 64,
        'prefix_length': 10
    }
    
    # Initialize model
    print(f"🔧 Initializing Vibe-tuned model...")
    model = VibeTunedBiomedicalEncoder(config=model_config)
    
    # Move to device
    model = model.to(device)
    
    print(f"✅ Model ready for training on {device}")
    print(f"📊 Teacher: {config['teacher']} → Student: {config['student']}\n")
    
    return model, device, config


# ==============================================================================
# MACBOOK PRO TRAINING RECOMMENDATIONS
# ==============================================================================

def get_macbook_training_config(config_name='medical_reasoning'):
    """
    Get optimized training hyperparameters for MacBook Pro
    
    Returns recommended batch size, learning rate, and other settings
    based on the chosen configuration and MacBook's memory.
    """
    
    config = VIBE_TUNING_CONFIGS[config_name]
    
    # Base recommendations for MacBook Pro (adjust based on your RAM)
    if 'ultra_light' in config_name or 'SmolLM2-135M' in config['student']:
        batch_size = 16
        gradient_accumulation = 2
        learning_rate = 5e-4
    elif 'efficient' in config_name or '1.7B' in config['student']:
        batch_size = 8
        gradient_accumulation = 4
        learning_rate = 3e-4
    elif 'production' in config_name or '1B' in config['student']:
        batch_size = 8
        gradient_accumulation = 4
        learning_rate = 3e-4
    elif '3B' in config['student'] or '4B' in config['student']:
        batch_size = 4
        gradient_accumulation = 8
        learning_rate = 2e-4
    else:
        batch_size = 4
        gradient_accumulation = 8
        learning_rate = 2e-4
    
    training_config = {
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation,
        'learning_rate': learning_rate,
        'num_epochs': 10,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'fp16': False,  # MPS doesn't support fp16 yet
        'use_mps': torch.backends.mps.is_available(),
        'num_workers': 0,  # Set to 0 for MPS compatibility
        'pin_memory': False  # Set to False for MPS
    }
    
    print("\n" + "="*80)
    print(f"⚙️  MacBook Pro Training Configuration")
    print("="*80)
    print(f"  Batch Size:                {training_config['batch_size']}")
    print(f"  Gradient Accumulation:     {training_config['gradient_accumulation_steps']}")
    print(f"  Effective Batch Size:      {training_config['batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"  Learning Rate:             {training_config['learning_rate']}")
    print(f"  Number of Epochs:          {training_config['num_epochs']}")
    print(f"  MPS (GPU) Enabled:         {training_config['use_mps']}")
    print("="*80 + "\n")
    
    return training_config


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "🚀 " + "="*76)
    print("  Vibe-Tuning Setup for Surgical Risk Prediction (MacBook Pro)")
    print("="*78 + "\n")
    
    # Example 1: Medical Reasoning (Recommended)
    print("📌 EXAMPLE 1: Medical Reasoning Configuration")
    model, device, config = setup_vibe_tuning(
        config_name='medical_reasoning',
        use_lora=True,
        use_adapters=True
    )
    training_config = get_macbook_training_config('medical_reasoning')
    
    print(f"\n💡 To use this configuration in your training:")
    print(f"   1. The Teacher model ({config['teacher']}) will generate synthetic training data")
    print(f"   2. The Student model ({config['student']}) will be fine-tuned on this data")
    print(f"   3. Training will take approximately {config['training_time']}")
    print(f"   4. Expected memory usage: {config['memory_req']}")
    
    # Show all available configurations
    print("\n\n" + "="*80)
    print("📋 All Available Configurations:")
    print("="*80)
    for name, cfg in VIBE_TUNING_CONFIGS.items():
        print(f"\n  🔹 {name.upper()}")
        print(f"     Teacher: {cfg['teacher']}")
        print(f"     Student: {cfg['student']}")
        print(f"     Memory:  {cfg['memory_req']}")
        print(f"     Time:    {cfg['training_time']}")
        print(f"     Use:     {cfg['use_case']}")
    
    print("\n\n" + "="*80)
    print("✅ Setup complete! You can now train with any configuration.")
    print("="*80 + "\n")
    
    # Recommend based on MacBook specs
    print("💻 MacBook Pro Recommendations:")
    print("   • 8 GB RAM:   Use 'ultra_light' or 'production'")
    print("   • 16 GB RAM:  Use 'efficient' or 'balanced'")
    print("   • 32+ GB RAM: Use 'medical_reasoning' (recommended)")
    print("\n")
