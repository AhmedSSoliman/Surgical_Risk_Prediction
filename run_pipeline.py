# run_pipeline.py
"""
Complete pipeline for surgical risk prediction
Runs: preprocessing -> training -> evaluation -> explainability
"""

import os
# Set environment variable to suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional  # Add this line
import pickle  # Add this import

# Import all modules
from config import *
from data.data_loader import MIMICDataLoader, SampleDataGenerator
from preprocessing import TimeSeriesPreprocessor, ClinicalNotesPreprocessor, ModalityAligner
from data.dataset import SurgicalRiskDataset, create_dataloaders, save_datasets, load_datasets
from models.model import MultimodalSurgicalRiskModel
from training.train import Trainer
from training.evaluate import Evaluator
from explainability import SHAPExplainer, AttentionVisualizer, FeatureImportanceAnalyzer
from visualization import ResultsPlotter, ExplainabilityPlotter
from utils import set_seed, get_device, count_parameters, create_logger



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Surgical Risk Prediction Pipeline')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['preprocess', 'train', 'evaluate', 'explain', 'full'],
                       help='Pipeline mode')
    
    parser.add_argument('--data_source', type=str, default='sample',
                       choices=['mimic', 'sample'],
                       help='Data source')
    
    parser.add_argument('--mimic_path', type=str, default=MIMIC_PATH,
                       help='Path to MIMIC-III dataset')
    
    parser.add_argument('--n_patients', type=int, default=1000,
                       help='Number of patients to process')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    
    parser.add_argument('--load_checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def preprocess_data(args, logger):
    """Preprocessing pipeline"""
    logger.info("="*80)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Initialize preprocessors
    ts_preprocessor = TimeSeriesPreprocessor()
    notes_preprocessor = ClinicalNotesPreprocessor()
    aligner = ModalityAligner()
    
    # Load data
    if args.data_source == 'mimic':
        logger.info(f"Loading MIMIC-III data from: {args.mimic_path}")
        loader = MIMICDataLoader(args.mimic_path)
        surgical_cohort = loader.identify_surgical_cohort(n_patients=args.n_patients)
    else:
        logger.info("Using sample data")
        surgical_cohort = None
    
    # Process each patient
    aligned_data_list = []
    
    n_process = args.n_patients if surgical_cohort is not None else 10
    
    logger.info(f"Processing {n_process} patients...")
    
    for i in tqdm(range(n_process), desc="Processing patients"):
        try:
            # Load patient data
            if args.data_source == 'mimic' and surgical_cohort is not None:
                hadm_id = surgical_cohort.iloc[i]['HADM_ID']
                patient_data = loader.load_patient_data(hadm_id)
            else:
                patient_data = SampleDataGenerator.generate_sample_patient()
            
            surgery_time = patient_data['surgery_time']
            
            # Preprocess labs (preop and intraop)
            labs_preop, labs_preop_names, labs_preop_meta = ts_preprocessor.preprocess_labs(
                patient_data['labs'],
                surgery_time,
                phase='preoperative'
            )
            
            labs_intraop, labs_intraop_names, labs_intraop_meta = ts_preprocessor.preprocess_labs(
                patient_data['labs'],
                surgery_time,
                phase='intraoperative'
            )
            
            # Preprocess vitals (preop and intraop)
            vitals_preop, vitals_preop_names, vitals_preop_meta = ts_preprocessor.preprocess_vitals(
                patient_data['vitals'],
                surgery_time,
                phase='preoperative'
            )
            
            vitals_intraop, vitals_intraop_names, vitals_intraop_meta = ts_preprocessor.preprocess_vitals(
                patient_data['vitals'],
                surgery_time,
                phase='intraoperative'
            )
            
            # Preprocess notes (preop and intraop separately)
            notes_preop = notes_preprocessor.preprocess_notes(
                patient_data['notes'],
                surgery_time,
                phase='preoperative'
            )
            
            notes_intraop = notes_preprocessor.preprocess_notes(
                patient_data['notes'],
                surgery_time,
                phase='intraoperative'
            )
            
            # Align all modalities
            aligned_data = aligner.align_all_modalities(
                labs_preop=(labs_preop, labs_preop_names, labs_preop_meta),
                labs_intraop=(labs_intraop, labs_intraop_names, labs_intraop_meta),
                vitals_preop=(vitals_preop, vitals_preop_names, vitals_preop_meta),
                vitals_intraop=(vitals_intraop, vitals_intraop_names, vitals_intraop_meta),
                notes_preop=notes_preop,
                notes_intraop=notes_intraop,
                static_features=patient_data['demographics'],
                medications_df=patient_data['medications'],
                outcomes=patient_data['outcomes']
            )
            
            aligned_data['hadm_id'] = patient_data['hadm_id']
            aligned_data_list.append(aligned_data)
            
        except Exception as e:
            logger.warning(f"Error processing patient {i}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(aligned_data_list)} patients")
    
    # Save preprocessed data
    save_path = DATA_DIR / 'aligned_data.pkl'
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(aligned_data_list, f)
    
    logger.info(f"Preprocessed data saved to: {save_path}")
    
    elapsed = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed/60:.1f} minutes")
    
    return aligned_data_list


def train_model(args, logger, aligned_data_list=None):
    """Training pipeline"""
    logger.info("="*80)
    logger.info("TRAINING PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Load preprocessed data if not provided
    if aligned_data_list is None:
        logger.info("Loading preprocessed data...")
        data_path = DATA_DIR / 'aligned_data.pkl'
        import pickle
        with open(data_path, 'rb') as f:
            aligned_data_list = pickle.load(f)
    
    # Check for empty data
    if not aligned_data_list or len(aligned_data_list) == 0:
        logger.error("No patient data was loaded or processed. Check your data source and preprocessing steps.")
        raise ValueError("No patient data available for training. Please check your data source and preprocessing.")
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        aligned_data_list,
        batch_size=args.batch_size,
        val_split=TRAINING_CONFIG['validation_split'],
        test_split=TRAINING_CONFIG['test_split'],
        random_seed=args.seed
    )
    
    # Check if train_loader is empty
    if len(train_loader) == 0:
        logger.error("Train loader is empty. Not enough data to create batches.")
        raise ValueError("Train loader is empty. Please increase n_patients or reduce batch_size.")
    
    # Get input dimensions from first batch
    sample_batch = next(iter(train_loader))
    ts_input_size = sample_batch['time_series_full'].shape[2]
    static_input_size = sample_batch['static'].shape[1]
    text_input_size = sample_batch['text_combined'].shape[1]
    
    logger.info(f"Input dimensions:")
    logger.info(f"  Time series: {ts_input_size}")
    logger.info(f"  Static features: {static_input_size}")
    logger.info(f"  Text embedding: {text_input_size}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MultimodalSurgicalRiskModel(
        ts_input_size=ts_input_size,
        static_input_size=static_input_size,
        text_input_size=text_input_size,
        config=MODEL_CONFIG
    )
    
    # Count parameters
    count_parameters(model)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAINING_CONFIG
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    
    # Plot training curves
    plotter = ResultsPlotter()
    plotter.plot_training_curves(history)
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed/60:.1f} minutes")
    
    return model, test_loader, history


def evaluate_model(args, logger, model=None, test_loader=None):
    """Evaluation pipeline"""
    logger.info("="*80)
    logger.info("EVALUATION PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Load model if not provided
    if model is None:
        logger.info("Loading model from checkpoint...")
        
        checkpoint_path = args.load_checkpoint or (MODEL_DIR / 'best_model.pt')
        checkpoint = torch.load(checkpoint_path)
        
        # Reconstruct model
        # Need to get dimensions from saved data or config
        logger.error("Model reconstruction not implemented. Please provide model.")
        return None
    
    # Load test data if not provided
    if test_loader is None:
        logger.info("Loading test data...")
        data_path = DATA_DIR / 'aligned_data.pkl'
        import pickle
        with open(data_path, 'rb') as f:
            aligned_data_list = pickle.load(f)
        
        _, _, test_loader = create_dataloaders(
            aligned_data_list,
            batch_size=TRAINING_CONFIG['batch_size']
        )
    
    # Evaluate
    logger.info("Running evaluation...")
    device = get_device()
    model.to(device)
    model.eval()
    
    # Collect predictions
    all_predictions = {task: [] for task in COMPLICATIONS.keys()}
    all_targets = {task: [] for task in COMPLICATIONS.keys()}
    all_uncertainties = {task: [] for task in COMPLICATIONS.keys()}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            outputs = model(
                time_series=batch['time_series_full'],
                phase_markers=batch['phase_markers'],
                ts_attention_mask=batch['mask_full'],
                text_embedding=batch['text_combined'],
                static_features=batch['static'],
                compute_uncertainty=True
            )
            
            targets = {task: batch['outcomes'][:, i] 
                     for i, task in enumerate(sorted(COMPLICATIONS.keys()))}
            
            for task in COMPLICATIONS.keys():
                all_predictions[task].append(outputs['predictions'][task].cpu())
                all_targets[task].append(targets[task].cpu())
                all_uncertainties[task].append(outputs['uncertainties'][task].cpu())
    
    # Concatenate
    predictions = {task: torch.cat(preds) for task, preds in all_predictions.items()}
    targets = {task: torch.cat(targs) for task, targs in all_targets.items()}
    uncertainties = {task: torch.cat(uncs) for task, uncs in all_uncertainties.items()}
    
    # Evaluate
    evaluator = Evaluator()
    results = evaluator.evaluate(predictions, targets, uncertainties)
    evaluator.print_results(results)
    
    # Create visualizations
    plotter = ResultsPlotter()
    plotter.plot_roc_curves(results)
    plotter.plot_pr_curves(results)
    plotter.plot_confusion_matrices(results)
    
    # Convert to numpy for calibration plots
    predictions_np = {k: v.numpy() for k, v in predictions.items()}
    targets_np = {k: v.numpy() for k, v in targets.items()}
    
    plotter.plot_calibration_curves(predictions_np, targets_np)
    plotter.plot_performance_summary(results)
    
    # Uncertainty analysis
    exp_plotter = ExplainabilityPlotter()
    uncertainties_np = {k: v.numpy() for k, v in uncertainties.items()}
    exp_plotter.plot_uncertainty_analysis(predictions_np, uncertainties_np, targets_np)
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed/60:.1f} minutes")
    
    return results


def explain_model(args, logger, model=None, test_loader=None):
    """Explainability pipeline"""
    logger.info("="*80)
    logger.info("EXPLAINABILITY PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Load model if not provided
    if model is None:
        logger.error("Model not provided. Cannot run explainability.")
        return None
    
    # Load test data if not provided
    if test_loader is None:
        logger.info("Loading test data...")
        data_path = DATA_DIR / 'aligned_data.pkl'
        import pickle
        with open(data_path, 'rb') as f:
            aligned_data_list = pickle.load(f)
        
        _, _, test_loader = create_dataloaders(
            aligned_data_list,
            batch_size=TRAINING_CONFIG['batch_size']
        )
    
    device = get_device()
    model.to(device)
    model.eval()
    
    # SHAP Explainer
    if EXPLAINABILITY_CONFIG['shap']['enabled']:
        logger.info("Running SHAP analysis...")
        shap_explainer = SHAPExplainer(model, test_loader, device)
        
        # Get test samples
        test_batch = next(iter(test_loader))
        
        shap_results = shap_explainer.explain_predictions(
            test_batch,
            feature_names=['Feature ' + str(i) for i in range(test_batch['static'].shape[1])]
        )
    
    # Attention Visualization
    if EXPLAINABILITY_CONFIG['attention']['enabled']:
        logger.info("Visualizing attention patterns...")
        attention_viz = AttentionVisualizer(model, device)
        
        test_batch = next(iter(test_loader))
        attention_results = attention_viz.visualize_attention(test_batch, sample_idx=0)
    
    # Feature Importance
    if EXPLAINABILITY_CONFIG['feature_importance']['enabled']:
        logger.info("Computing feature importance...")
        importance_analyzer = FeatureImportanceAnalyzer(model, test_loader, device)
        
        importance_scores = importance_analyzer.compute_permutation_importance()
    
    # Explainability plots
    exp_plotter = ExplainabilityPlotter()
    
    # Get sample for visualization
    test_batch = next(iter(test_loader))
    sample_ts = test_batch['time_series_full'][0].numpy()
    sample_predictions = {}
    
    with torch.no_grad():
        outputs = model(
            time_series=test_batch['time_series_full'][:1].to(device),
            phase_markers=test_batch['phase_markers'][:1].to(device),
            ts_attention_mask=test_batch['mask_full'][:1].to(device),
            text_embedding=test_batch['text_combined'][:1].to(device),
            static_features=test_batch['static'][:1].to(device)
        )
        
        for task in COMPLICATIONS.keys():
            sample_predictions[task] = outputs['predictions'][task].cpu().numpy()[0]
    
    # Plot temporal dynamics
    exp_plotter.plot_temporal_dynamics(
        sample_ts,
        sample_predictions,
        ['Feature ' + str(i) for i in range(sample_ts.shape[1])]
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Explainability analysis completed in {elapsed/60:.1f} minutes")


def run_full_pipeline(args, logger):
    """Run complete pipeline"""
    logger.info("="*80)
    logger.info("FULL PIPELINE - SURGICAL RISK PREDICTION")
    logger.info("="*80)
    
    pipeline_start = time.time()
    
    # Step 1: Preprocessing
    logger.info("\n[STEP 1/4] Preprocessing...")
    aligned_data_list = preprocess_data(args, logger)
    
    # Step 2: Training
    logger.info("\n[STEP 2/4] Training...")
    model, test_loader, history = train_model(args, logger, aligned_data_list)
    
    # Step 3: Evaluation
    logger.info("\n[STEP 3/4] Evaluation...")
    results = evaluate_model(args, logger, model, test_loader)
    
    # Step 4: Explainability
    logger.info("\n[STEP 4/4] Explainability...")
    explain_model(args, logger, model, test_loader)
    
    # Final summary
    total_elapsed = time.time() - pipeline_start
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info(f"Models saved to: {MODEL_DIR}")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create logger
    logger = create_logger(
        'surgical_risk_prediction',
        log_file=f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logger.info(f"Starting pipeline in mode: {args.mode}")
    logger.info(f"Configuration:")
    logger.info(f"  Data source: {args.data_source}")
    logger.info(f"  N patients: {args.n_patients}")
    logger.info(f"  Random seed: {args.seed}")
    
    # Run pipeline based on mode
    if args.mode == 'preprocess':
        preprocess_data(args, logger)
    
    elif args.mode == 'train':
        train_model(args, logger)
    
    elif args.mode == 'evaluate':
        evaluate_model(args, logger)
    
    elif args.mode == 'explain':
        explain_model(args, logger)
    
    elif args.mode == 'full':
        run_full_pipeline(args, logger)
    
    logger.info("\nPipeline finished successfully!")


if __name__ == '__main__':
    main()