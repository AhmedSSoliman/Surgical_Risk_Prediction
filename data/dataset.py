# data/dataset.py
"""
PyTorch Dataset for Multimodal Surgical Risk Prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

from config import TRAINING_CONFIG, COMPLICATIONS

class SurgicalRiskDataset(Dataset):
    """
    PyTorch Dataset for multimodal surgical risk prediction
    
    Returns aligned multimodal data:
    - Time series (preop + intraop)
    - Text embeddings (preop + intraop)
    - Static features
    - Medication features
    - Cross-modal features
    - Outcomes (9 complications)
    """
    
    def __init__(self, 
                 aligned_data_list: List[Dict],
                 transform=None):
        """
        Args:
            aligned_data_list: List of aligned data dictionaries
            transform: Optional data augmentation transform
        """
        self.data = aligned_data_list
        self.transform = transform
        
        # Validate data
        self._validate_data()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            Dictionary with tensors for each modality
        """
        sample_data = self.data[idx]
        
        # Time series
        ts_full = torch.FloatTensor(sample_data['time_series']['full_sequence'])
        ts_preop = torch.FloatTensor(sample_data['time_series']['preoperative']['combined'])
        ts_intraop = torch.FloatTensor(sample_data['time_series']['intraoperative']['combined'])
        phase_markers = torch.FloatTensor(sample_data['time_series']['phase_markers'])
        
        # Text embeddings
        text_preop = torch.FloatTensor(sample_data['text']['preoperative']['embedding'])
        text_intraop = torch.FloatTensor(sample_data['text']['intraoperative']['embedding'])
        text_combined = torch.FloatTensor(sample_data['text']['combined_embedding'])
        
        # Static features
        static = torch.FloatTensor(sample_data['static']['array'])
        
        # Medication features
        medications = torch.FloatTensor(sample_data['medications']['array'])
        
        # Cross-modal features
        cross_modal = torch.FloatTensor(sample_data['cross_modal_features']['array'])
        
        # Attention masks
        mask_preop = torch.FloatTensor(sample_data['attention_masks']['preoperative'])
        mask_intraop = torch.FloatTensor(sample_data['attention_masks']['intraoperative'])
        mask_full = torch.FloatTensor(sample_data['attention_masks']['full_sequence'])
        
        # Outcomes
        outcomes = torch.FloatTensor(sample_data['outcomes']['array'])
        
        # Create sample dictionary
        sample = {
            # Time series
            'time_series_full': ts_full,
            'time_series_preop': ts_preop,
            'time_series_intraop': ts_intraop,
            'phase_markers': phase_markers,
            
            # Text
            'text_preop': text_preop,
            'text_intraop': text_intraop,
            'text_combined': text_combined,
            
            # Other modalities
            'static': static,
            'medications': medications,
            'cross_modal': cross_modal,
            
            # Masks
            'mask_preop': mask_preop,
            'mask_intraop': mask_intraop,
            'mask_full': mask_full,
            
            # Outcomes
            'outcomes': outcomes,
            
            # Metadata (not tensors)
            'hadm_id': sample_data.get('hadm_id', 0)
        }
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _validate_data(self):
        """Validate that all samples have consistent shapes and pad if necessary"""
        if not self.data:
            raise ValueError("Dataset is empty")
        
        # Find maximum dimensions across all samples
        max_ts_full_len = max(d['time_series']['full_sequence'].shape[0] for d in self.data)
        max_ts_full_features = max(d['time_series']['full_sequence'].shape[1] for d in self.data)
        
        max_ts_preop_len = max(d['time_series']['preoperative']['combined'].shape[0] for d in self.data)
        max_ts_preop_features = max(d['time_series']['preoperative']['combined'].shape[1] for d in self.data)
        
        max_ts_intraop_len = max(d['time_series']['intraoperative']['combined'].shape[0] for d in self.data)
        max_ts_intraop_features = max(d['time_series']['intraoperative']['combined'].shape[1] for d in self.data)
        
        max_text_preop_dim = max(d['text']['preoperative']['embedding'].shape[0] for d in self.data)
        max_text_intraop_dim = max(d['text']['intraoperative']['embedding'].shape[0] for d in self.data)
        max_text_combined_dim = max(d['text']['combined_embedding'].shape[0] for d in self.data)
        
        max_static_dim = max(d['static']['array'].shape[0] for d in self.data)
        max_med_dim = max(d['medications']['array'].shape[0] for d in self.data)
        max_cross_modal_dim = max(d['cross_modal_features']['array'].shape[0] for d in self.data)
        
        max_phase_markers = max(d['time_series']['phase_markers'].shape[0] for d in self.data)
        max_mask_preop = max(d['attention_masks']['preoperative'].shape[0] for d in self.data)
        max_mask_intraop = max(d['attention_masks']['intraoperative'].shape[0] for d in self.data)
        max_mask_full = max(d['attention_masks']['full_sequence'].shape[0] for d in self.data)
        
        # Pad all samples to maximum dimensions
        for sample in self.data:
            # Pad time series full sequence
            ts_full = sample['time_series']['full_sequence']
            if ts_full.shape != (max_ts_full_len, max_ts_full_features):
                padded = np.zeros((max_ts_full_len, max_ts_full_features))
                padded[:ts_full.shape[0], :ts_full.shape[1]] = ts_full
                sample['time_series']['full_sequence'] = padded
            
            # Pad preoperative combined
            ts_preop = sample['time_series']['preoperative']['combined']
            if ts_preop.shape != (max_ts_preop_len, max_ts_preop_features):
                padded = np.zeros((max_ts_preop_len, max_ts_preop_features))
                padded[:ts_preop.shape[0], :ts_preop.shape[1]] = ts_preop
                sample['time_series']['preoperative']['combined'] = padded
            
            # Pad intraoperative combined
            ts_intraop = sample['time_series']['intraoperative']['combined']
            if ts_intraop.shape != (max_ts_intraop_len, max_ts_intraop_features):
                padded = np.zeros((max_ts_intraop_len, max_ts_intraop_features))
                padded[:ts_intraop.shape[0], :ts_intraop.shape[1]] = ts_intraop
                sample['time_series']['intraoperative']['combined'] = padded
            
            # Pad phase markers
            phase_markers = sample['time_series']['phase_markers']
            if phase_markers.shape[0] < max_phase_markers:
                sample['time_series']['phase_markers'] = np.pad(phase_markers, (0, max_phase_markers - phase_markers.shape[0]))
            
            # Pad text embeddings
            text_preop = sample['text']['preoperative']['embedding']
            if text_preop.shape[0] < max_text_preop_dim:
                sample['text']['preoperative']['embedding'] = np.pad(text_preop, (0, max_text_preop_dim - text_preop.shape[0]))
            
            text_intraop = sample['text']['intraoperative']['embedding']
            if text_intraop.shape[0] < max_text_intraop_dim:
                sample['text']['intraoperative']['embedding'] = np.pad(text_intraop, (0, max_text_intraop_dim - text_intraop.shape[0]))
            
            text_combined = sample['text']['combined_embedding']
            if text_combined.shape[0] < max_text_combined_dim:
                sample['text']['combined_embedding'] = np.pad(text_combined, (0, max_text_combined_dim - text_combined.shape[0]))
            
            # Pad static, medications, cross-modal
            static = sample['static']['array']
            if static.shape[0] < max_static_dim:
                sample['static']['array'] = np.pad(static, (0, max_static_dim - static.shape[0]))
            
            meds = sample['medications']['array']
            if meds.shape[0] < max_med_dim:
                sample['medications']['array'] = np.pad(meds, (0, max_med_dim - meds.shape[0]))
            
            cross_modal = sample['cross_modal_features']['array']
            if cross_modal.shape[0] < max_cross_modal_dim:
                sample['cross_modal_features']['array'] = np.pad(cross_modal, (0, max_cross_modal_dim - cross_modal.shape[0]))
            
            # Pad attention masks
            mask_preop = sample['attention_masks']['preoperative']
            if mask_preop.shape[0] < max_mask_preop:
                sample['attention_masks']['preoperative'] = np.pad(mask_preop, (0, max_mask_preop - mask_preop.shape[0]))
            
            mask_intraop = sample['attention_masks']['intraoperative']
            if mask_intraop.shape[0] < max_mask_intraop:
                sample['attention_masks']['intraoperative'] = np.pad(mask_intraop, (0, max_mask_intraop - mask_intraop.shape[0]))
            
            mask_full = sample['attention_masks']['full_sequence']
            if mask_full.shape[0] < max_mask_full:
                sample['attention_masks']['full_sequence'] = np.pad(mask_full, (0, max_mask_full - mask_full.shape[0]))
        
        # Store final shapes
        self.ts_shape = (max_ts_full_len, max_ts_full_features)
        self.text_dim = max_text_combined_dim
        self.static_dim = max_static_dim
        self.med_dim = max_med_dim
        self.cross_modal_dim = max_cross_modal_dim
        
        print(f"Dataset validated: {len(self.data)} samples")
        print(f"  Time series shape: {self.ts_shape}")
        print(f"  Text embedding dim: {self.text_dim}")
        print(f"  Static features: {self.static_dim}")
        print(f"  Medication features: {self.med_dim}")
        print(f"  Cross-modal features: {self.cross_modal_dim}")
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Calculate class weights for imbalanced data"""
        outcomes_array = np.array([d['outcomes']['array'] for d in self.data])
        
        weights = {}
        for i, comp_name in enumerate(sorted(COMPLICATIONS.keys())):
            pos_count = outcomes_array[:, i].sum()
            neg_count = len(outcomes_array) - pos_count
            
            if pos_count > 0:
                pos_weight = neg_count / pos_count
            else:
                pos_weight = 1.0
            
            weights[comp_name] = torch.tensor([1.0, pos_weight])
        
        return weights
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        outcomes_array = np.array([d['outcomes']['array'] for d in self.data])
        
        stats = {
            'num_samples': len(self.data),
            'complication_prevalence': {}
        }
        
        for i, comp_name in enumerate(sorted(COMPLICATIONS.keys())):
            prevalence = outcomes_array[:, i].mean()
            stats['complication_prevalence'][comp_name] = float(prevalence)
        
        return stats


def _normalize_all_samples(data_list: List[Dict]) -> List[Dict]:
    """
    Normalize all samples to have consistent dimensions across entire dataset.
    Must be called BEFORE splitting into train/val/test to ensure all splits have same shape.
    """
    if not data_list:
        return data_list
    
    # Find maximum dimensions across ALL samples
    max_ts_full_len = max(d['time_series']['full_sequence'].shape[0] for d in data_list)
    max_ts_full_features = max(d['time_series']['full_sequence'].shape[1] for d in data_list)
    
    max_ts_preop_len = max(d['time_series']['preoperative']['combined'].shape[0] for d in data_list)
    max_ts_preop_features = max(d['time_series']['preoperative']['combined'].shape[1] for d in data_list)
    
    max_ts_intraop_len = max(d['time_series']['intraoperative']['combined'].shape[0] for d in data_list)
    max_ts_intraop_features = max(d['time_series']['intraoperative']['combined'].shape[1] for d in data_list)
    
    max_text_preop_dim = max(d['text']['preoperative']['embedding'].shape[0] for d in data_list)
    max_text_intraop_dim = max(d['text']['intraoperative']['embedding'].shape[0] for d in data_list)
    max_text_combined_dim = max(d['text']['combined_embedding'].shape[0] for d in data_list)
    
    max_static_dim = max(d['static']['array'].shape[0] for d in data_list)
    max_med_dim = max(d['medications']['array'].shape[0] for d in data_list)
    max_cross_modal_dim = max(d['cross_modal_features']['array'].shape[0] for d in data_list)
    
    max_phase_markers = max(d['time_series']['phase_markers'].shape[0] for d in data_list)
    max_mask_preop = max(d['attention_masks']['preoperative'].shape[0] for d in data_list)
    max_mask_intraop = max(d['attention_masks']['intraoperative'].shape[0] for d in data_list)
    max_mask_full = max(d['attention_masks']['full_sequence'].shape[0] for d in data_list)
    
    print(f"  Maximum dimensions found:")
    print(f"    Time series: ({max_ts_full_len}, {max_ts_full_features})")
    print(f"    Text: {max_text_combined_dim}, Static: {max_static_dim}")
    
    # Pad all samples to maximum dimensions
    for sample in data_list:
        # Pad time series arrays
        _pad_2d_array(sample['time_series'], 'full_sequence', max_ts_full_len, max_ts_full_features)
        _pad_2d_array(sample['time_series']['preoperative'], 'combined', max_ts_preop_len, max_ts_preop_features)
        _pad_2d_array(sample['time_series']['intraoperative'], 'combined', max_ts_intraop_len, max_ts_intraop_features)
        
        # Pad 1D arrays
        _pad_1d_array(sample['time_series'], 'phase_markers', max_phase_markers)
        _pad_1d_array(sample['text']['preoperative'], 'embedding', max_text_preop_dim)
        _pad_1d_array(sample['text']['intraoperative'], 'embedding', max_text_intraop_dim)
        _pad_1d_array(sample['text'], 'combined_embedding', max_text_combined_dim)
        _pad_1d_array(sample['static'], 'array', max_static_dim)
        _pad_1d_array(sample['medications'], 'array', max_med_dim)
        _pad_1d_array(sample['cross_modal_features'], 'array', max_cross_modal_dim)
        _pad_1d_array(sample['attention_masks'], 'preoperative', max_mask_preop)
        _pad_1d_array(sample['attention_masks'], 'intraoperative', max_mask_intraop)
        _pad_1d_array(sample['attention_masks'], 'full_sequence', max_mask_full)
    
    print(f"  All {len(data_list)} samples normalized to consistent dimensions")
    return data_list


def _pad_2d_array(container: Dict, key: str, max_rows: int, max_cols: int):
    """Pad a 2D array to specified dimensions"""
    arr = container[key]
    if arr.shape != (max_rows, max_cols):
        padded = np.zeros((max_rows, max_cols))
        padded[:arr.shape[0], :arr.shape[1]] = arr
        container[key] = padded


def _pad_1d_array(container: Dict, key: str, max_len: int):
    """Pad a 1D array to specified length"""
    arr = container[key]
    if arr.shape[0] < max_len:
        container[key] = np.pad(arr, (0, max_len - arr.shape[0]))


def create_dataloaders(aligned_data_list: List[Dict],
                      batch_size: int = None,
                      val_split: float = None,
                      test_split: float = None,
                      random_seed: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        aligned_data_list: List of aligned data dictionaries
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        random_seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    val_split = val_split or TRAINING_CONFIG['validation_split']
    test_split = test_split or TRAINING_CONFIG['test_split']
    random_seed = random_seed or TRAINING_CONFIG['random_seed']
    
    if not aligned_data_list or len(aligned_data_list) == 0:
        print("Warning: No data provided to create_dataloaders. Returning empty loaders.")
        raise ValueError("No data provided to create_dataloaders.")
    
    # CRITICAL: Normalize all data BEFORE splitting to ensure consistent shapes
    print("Normalizing all samples to consistent dimensions...")
    aligned_data_list = _normalize_all_samples(aligned_data_list)
    
    # Split data with stratification for better handling of rare complications
    # Handle small datasets: if less than 3 samples, use all for training, skip splitting
    if len(aligned_data_list) < 3:
        train_data = aligned_data_list
        val_data = []
        test_data = []
    else:
        # Create stratification label based on most common complication
        # This ensures each split has at least some positive cases
        stratify_labels = []
        for data in aligned_data_list:
            outcomes = data['outcomes']['array']
            # Use first complication with positive case, or -1 if all negative
            label = -1
            for i, val in enumerate(outcomes):
                if val > 0.5:  # Positive case
                    label = i
                    break
            stratify_labels.append(label)
        
        stratify_labels = np.array(stratify_labels)
        
        # Check if we have enough samples per class for stratification
        unique_labels, counts = np.unique(stratify_labels, return_counts=True)
        min_count = counts.min()
        
        use_stratify = min_count >= 2  # Need at least 2 samples per class for stratified split
        
        if use_stratify:
            print(f"Using stratified splitting to ensure balanced complication distribution")
            train_val_data, test_data = train_test_split(
                aligned_data_list,
                test_size=test_split,
                random_state=random_seed,
                stratify=stratify_labels
            )
            
            # Re-compute stratify labels for train/val split
            train_val_indices = [aligned_data_list.index(d) for d in train_val_data]
            stratify_train_val = stratify_labels[train_val_indices]
            
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_split / (1 - test_split),
                random_state=random_seed,
                stratify=stratify_train_val
            )
        else:
            print(f"Warning: Not enough samples per class for stratified split, using random split")
            train_val_data, test_data = train_test_split(
                aligned_data_list,
                test_size=test_split,
                random_state=random_seed
            )
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_split / (1 - test_split),
                random_state=random_seed
            )
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Adjust batch_size if dataset is smaller
    effective_batch_size = min(batch_size, len(train_data))
    if effective_batch_size < batch_size:
        print(f"Warning: Reducing batch_size from {batch_size} to {effective_batch_size} due to small dataset")
    
    # Create datasets
    train_dataset = SurgicalRiskDataset(train_data)
    val_dataset = SurgicalRiskDataset(val_data) if val_data else SurgicalRiskDataset([])
    test_dataset = SurgicalRiskDataset(test_data) if test_data else SurgicalRiskDataset([])
    
    # Create dataloaders (drop_last=False for small datasets)
    drop_last_train = len(train_data) > effective_batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0 if len(train_data) < 10 else 4,
        pin_memory=True,
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0 if len(val_data) < 10 else 4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0 if len(test_data) < 10 else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def save_datasets(train_data: List[Dict],
                 val_data: List[Dict],
                 test_data: List[Dict],
                 output_dir: Path):
    """Save preprocessed datasets to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(output_dir / 'val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    with open(output_dir / 'test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"Datasets saved to {output_dir}")


def load_datasets(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load preprocessed datasets from disk"""
    data_dir = Path(data_dir)
    
    with open(data_dir / 'train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(data_dir / 'val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    with open(data_dir / 'test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Datasets loaded from {data_dir}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data