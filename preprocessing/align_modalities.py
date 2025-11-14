# preprocessing/align_modalities.py
"""
Multimodal Data Alignment with Temporal Synchronization
Aligns time series, notes, and static features to common reference
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import TEMPORAL_WINDOWS

class ModalityAligner:
    """
    Aligns multiple data modalities to common temporal reference
    
    Modalities:
    - Time series (labs, vitals) - preoperative & intraoperative
    - Clinical notes (text) - phase-specific
    - Static features (demographics, comorbidities)
    - Medications (time-varying)
    - Outcomes (binary labels)
    """
    
    def __init__(self, reference_time: pd.Timestamp = None):
        self.reference_time = reference_time
        
    def align_all_modalities(self,
                            labs_preop: Tuple[np.ndarray, List[str], Dict],
                            labs_intraop: Tuple[np.ndarray, List[str], Dict],
                            vitals_preop: Tuple[np.ndarray, List[str], Dict],
                            vitals_intraop: Tuple[np.ndarray, List[str], Dict],
                            notes_preop: Dict,
                            notes_intraop: Dict,
                            static_features: Dict,
                            medications_df: pd.DataFrame = None,
                            outcomes: Dict = None) -> Dict:
        """
        Align all data modalities with temporal separation
        
        Args:
            labs_preop: Preoperative labs (array, names, metadata)
            labs_intraop: Intraoperative labs
            vitals_preop: Preoperative vitals
            vitals_intraop: Intraoperative vitals
            notes_preop: Preoperative notes dict
            notes_intraop: Intraoperative notes dict
            static_features: Static patient features
            medications_df: Medication administration data
            outcomes: Outcome labels for 9 complications
            
        Returns:
            Dictionary with aligned multimodal data
        """
        aligned_data = {
            'time_series': {},
            'text': {},
            'static': {},
            'medications': {},
            'outcomes': {},
            'attention_masks': {},
            'metadata': {}
        }
        
        # 1. Align time series by phase
        aligned_data['time_series'] = self._align_time_series_multimodal(
            labs_preop, labs_intraop,
            vitals_preop, vitals_intraop
        )
        
        # 2. Process text data by phase
        aligned_data['text'] = self._align_text_modalities(
            notes_preop, notes_intraop
        )
        
        # 3. Static features
        aligned_data['static'] = self._process_static_features(static_features)
        
        # 4. Medication features
        if medications_df is not None and not medications_df.empty:
            aligned_data['medications'] = self._process_medications(medications_df)
        else:
            aligned_data['medications'] = self._empty_medication_features()
        
        # 5. Outcomes
        if outcomes:
            aligned_data['outcomes'] = self._process_outcomes(outcomes)
        else:
            aligned_data['outcomes'] = self._empty_outcomes()
        
        # 6. Create attention masks
        aligned_data['attention_masks'] = self._create_attention_masks(
            aligned_data['time_series']
        )
        
        # 7. Create cross-modal features
        aligned_data['cross_modal_features'] = self._create_cross_modal_features(
            aligned_data
        )
        
        # 8. Metadata
        aligned_data['metadata'] = self._generate_alignment_metadata(aligned_data)
        
        return aligned_data
    
    def _align_time_series_multimodal(self,
                                     labs_preop: Tuple,
                                     labs_intraop: Tuple,
                                     vitals_preop: Tuple,
                                     vitals_intraop: Tuple) -> Dict:
        """
        Align multimodal time series data
        
        Creates separate sequences for preop and intraop,
        then combines them with phase markers
        """
        # Extract arrays
        labs_preop_array, labs_preop_names, labs_preop_meta = labs_preop
        labs_intraop_array, labs_intraop_names, labs_intraop_meta = labs_intraop
        vitals_preop_array, vitals_preop_names, vitals_preop_meta = vitals_preop
        vitals_intraop_array, vitals_intraop_names, vitals_intraop_meta = vitals_intraop
        
        # Ensure same sequence length
        max_len = max(
            labs_preop_array.shape[0],
            labs_intraop_array.shape[0],
            vitals_preop_array.shape[0],
            vitals_intraop_array.shape[0]
        )
        
        # Pad all sequences
        labs_preop_padded = self._pad_sequence(labs_preop_array, max_len)
        labs_intraop_padded = self._pad_sequence(labs_intraop_array, max_len)
        vitals_preop_padded = self._pad_sequence(vitals_preop_array, max_len)
        vitals_intraop_padded = self._pad_sequence(vitals_intraop_array, max_len)
        
        # Pad arrays so all have the same number of columns before concatenation
        def pad_to_match(arrays):
            max_cols = max(arr.shape[1] for arr in arrays)
            padded = []
            for arr in arrays:
                if arr.shape[1] < max_cols:
                    pad_width = ((0,0),(0,max_cols-arr.shape[1]))
                    arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                padded.append(arr)
            return padded

        labs_preop_padded, vitals_preop_padded = pad_to_match([labs_preop_padded, vitals_preop_padded])
        labs_intraop_padded, vitals_intraop_padded = pad_to_match([labs_intraop_padded, vitals_intraop_padded])

        # Combine preoperative time series
        preop_combined = np.hstack([labs_preop_padded, vitals_preop_padded])

        # Combine intraoperative time series
        intraop_combined = np.hstack([labs_intraop_padded, vitals_intraop_padded])

        # Pad combined arrays before vstack
        preop_combined, intraop_combined = pad_to_match([preop_combined, intraop_combined])
        # Create concatenated sequence with phase markers
        # Shape: (2 * max_len, num_features)
        combined_sequence = np.vstack([preop_combined, intraop_combined])
        
        # Create phase markers (0 = preop, 1 = intraop)
        phase_markers = np.concatenate([
            np.zeros(max_len),
            np.ones(max_len)
        ])
        
        return {
            'preoperative': {
                'labs': labs_preop_padded,
                'vitals': vitals_preop_padded,
                'combined': preop_combined,
                'metadata': {
                    'labs': labs_preop_meta,
                    'vitals': vitals_preop_meta
                }
            },
            'intraoperative': {
                'labs': labs_intraop_padded,
                'vitals': vitals_intraop_padded,
                'combined': intraop_combined,
                'metadata': {
                    'labs': labs_intraop_meta,
                    'vitals': vitals_intraop_meta
                }
            },
            'full_sequence': combined_sequence,
            'phase_markers': phase_markers,
            'sequence_length': combined_sequence.shape[0],
            'num_features': combined_sequence.shape[1],
            'feature_names': labs_preop_names + vitals_preop_names
        }
    
    def _align_text_modalities(self,
                               notes_preop: Dict,
                               notes_intraop: Dict) -> Dict:
        """Align text embeddings by phase"""
        return {
            'preoperative': {
                'embedding': notes_preop['aggregated_embeddings'].get('preoperative', np.zeros(768)),
                'complication_mentions': notes_preop.get('complication_mentions', {}),
                'severity_score': notes_preop['metadata'].get('severity_score', 0.0),
                'num_notes': notes_preop['metadata'].get('num_notes', 0),
                'section_embeddings': notes_preop.get('section_embeddings', {})
            },
            'intraoperative': {
                'embedding': notes_intraop['aggregated_embeddings'].get('intraoperative', np.zeros(768)),
                'complication_mentions': notes_intraop.get('complication_mentions', {}),
                'severity_score': notes_intraop['metadata'].get('severity_score', 0.0),
                'num_notes': notes_intraop['metadata'].get('num_notes', 0),
                'section_embeddings': notes_intraop.get('section_embeddings', {})
            },
            'combined_embedding': self._combine_embeddings(
                notes_preop['aggregated_embeddings'].get('all', np.zeros(768)),
                notes_intraop['aggregated_embeddings'].get('all', np.zeros(768))
            ),
            'total_complication_mentions': self._merge_complication_mentions(
                notes_preop.get('complication_mentions', {}),
                notes_intraop.get('complication_mentions', {})
            )
        }
    
    def _process_static_features(self, features: Dict) -> Dict:
        """Process and encode static features"""
        feature_array = []
        feature_names = []
        
        # Demographics
        age = features.get('age', 50)
        feature_array.append(age / 100)  # Normalize
        feature_names.append('age_normalized')
        
        gender = features.get('gender', 'M')
        feature_array.append(1 if gender == 'M' else 0)
        feature_names.append('gender_male')
        
        # Admission type
        admission_type = features.get('admission_type', 'ELECTIVE')
        feature_array.append(1 if admission_type == 'EMERGENCY' else 0)
        feature_names.append('emergency_admission')
        
        # Comorbidities (if present)
        comorbidity_keys = [k for k in features.keys() if k.startswith('comorbid_')]
        for key in sorted(comorbidity_keys):
            feature_array.append(float(features[key]))
            feature_names.append(key)
        
        # Other numeric features
        for key, value in features.items():
            if key not in ['age', 'gender', 'admission_type', 'subject_id', 'hadm_id'] \
               and not key.startswith('comorbid_'):
                if isinstance(value, (int, float)):
                    feature_array.append(float(value))
                    feature_names.append(key)
                elif isinstance(value, bool):
                    feature_array.append(1.0 if value else 0.0)
                    feature_names.append(key)
        
        return {
            'array': np.array(feature_array),
            'names': feature_names,
            'raw': features
        }
    
    def _process_medications(self, medications_df: pd.DataFrame) -> Dict:
        """Process medication data into features"""
        if medications_df.empty:
            return self._empty_medication_features()
        
        # Medication categories
        med_categories = {
            'antibiotic': ['cillin', 'mycin', 'cycline', 'cef', 'vanc', 'metro'],
            'anticoagulant': ['heparin', 'warfarin', 'enoxaparin', 'rivaroxaban'],
            'vasopressor': ['epinephrine', 'norepinephrine', 'dopamine', 'vasopressin'],
            'analgesic': ['morphine', 'fentanyl', 'hydromorphone', 'oxycodone'],
            'sedative': ['propofol', 'midazolam', 'dexmedetomidine'],
            'insulin': ['insulin'],
            'steroid': ['prednisone', 'methylprednisolone', 'hydrocortisone'],
            'antihypertensive': ['metoprolol', 'atenolol', 'lisinopril', 'amlodipine'],
            'diuretic': ['furosemide', 'lasix', 'hydrochlorothiazide'],
            'beta_blocker': ['metoprolol', 'atenolol', 'carvedilol']
        }
        
        # Count medications by category
        category_counts = {cat: 0 for cat in med_categories.keys()}
        
        for _, row in medications_df.iterrows():
            drug = str(row.get('DRUG', '')).lower()
            
            for category, keywords in med_categories.items():
                if any(kw in drug for kw in keywords):
                    category_counts[category] += 1
        
        # Create feature array
        feature_array = list(category_counts.values())
        
        # Add total count
        feature_array.append(len(medications_df))
        
        # Add unique drug count
        unique_drugs = medications_df['DRUG'].nunique() if 'DRUG' in medications_df.columns else 0
        feature_array.append(unique_drugs)
        
        return {
            'array': np.array(feature_array),
            'names': list(med_categories.keys()) + ['total_medications', 'unique_drugs'],
            'category_counts': category_counts,
            'total_count': len(medications_df),
            'unique_count': unique_drugs
        }
    
    def _process_outcomes(self, outcomes: Dict) -> Dict:
        """Process outcome labels"""
        from config import COMPLICATIONS
        
        outcome_array = []
        outcome_names = []
        
        for comp_key in sorted(COMPLICATIONS.keys()):
            value = outcomes.get(comp_key, 0)
            outcome_array.append(int(value))
            outcome_names.append(comp_key)
        
        return {
            'array': np.array(outcome_array),
            'names': outcome_names,
            'raw': outcomes
        }
    
    def _create_attention_masks(self, time_series: Dict) -> Dict:
        """Create attention masks for valid time steps"""
        masks = {}
        
        # Preoperative mask
        preop_combined = time_series['preoperative']['combined']
        preop_mask = (np.sum(np.abs(preop_combined), axis=1) > 0).astype(float)
        masks['preoperative'] = preop_mask
        
        # Intraoperative mask
        intraop_combined = time_series['intraoperative']['combined']
        intraop_mask = (np.sum(np.abs(intraop_combined), axis=1) > 0).astype(float)
        masks['intraoperative'] = intraop_mask
        
        # Full sequence mask
        full_mask = np.concatenate([preop_mask, intraop_mask])
        masks['full_sequence'] = full_mask
        
        return masks
    
    def _create_cross_modal_features(self, aligned_data: Dict) -> Dict:
        """Create interaction features across modalities"""
        features = {}
        
        # Time series statistics
        ts_full = aligned_data['time_series']['full_sequence']
        features['ts_mean'] = float(np.mean(ts_full))
        features['ts_std'] = float(np.std(ts_full))
        features['ts_max'] = float(np.max(ts_full))
        features['ts_min'] = float(np.min(ts_full))
        features['ts_range'] = features['ts_max'] - features['ts_min']
        
        # Text features
        text_data = aligned_data['text']
        features['preop_severity'] = text_data['preoperative']['severity_score']
        features['intraop_severity'] = text_data['intraoperative']['severity_score']
        features['total_complication_mentions'] = sum(
            v['count'] for v in text_data['total_complication_mentions'].values()
        )
        
        # Medication-time series interaction
        if 'array' in aligned_data['medications']:
            med_count = aligned_data['medications']['total_count']
            ts_variability = np.std(ts_full)
            features['med_ts_interaction'] = med_count * ts_variability
        
        # Static-text interaction
        static_age = aligned_data['static']['array'][0] * 100  # Denormalize
        features['age_severity_interaction'] = static_age * features['preop_severity']
        
        # Create feature array
        feature_array = np.array(list(features.values()))
        feature_names = list(features.keys())
        
        return {
            'array': feature_array,
            'names': feature_names,
            'dict': features
        }
    
    def _combine_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Combine two embeddings (average pooling)"""
        if emb1.shape != emb2.shape:
            # Handle different shapes
            max_dim = max(emb1.shape[0], emb2.shape[0])
            emb1_padded = np.pad(emb1, (0, max_dim - emb1.shape[0]))
            emb2_padded = np.pad(emb2, (0, max_dim - emb2.shape[0]))
            return (emb1_padded + emb2_padded) / 2
        
        return (emb1 + emb2) / 2
    
    def _merge_complication_mentions(self, mentions1: Dict, mentions2: Dict) -> Dict:
        """Merge complication mention dictionaries"""
        merged = {}
        
        all_keys = set(mentions1.keys()) | set(mentions2.keys())
        
        for key in all_keys:
            count1 = mentions1.get(key, {}).get('count', 0)
            count2 = mentions2.get(key, {}).get('count', 0)
            
            contexts1 = mentions1.get(key, {}).get('contexts', [])
            contexts2 = mentions2.get(key, {}).get('contexts', [])
            
            merged[key] = {
                'count': count1 + count2,
                'contexts': contexts1 + contexts2
            }
        
        return merged
    
    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pad sequence to target length"""
        current_length = sequence.shape[0]
        
        if current_length < target_length:
            padding = np.zeros((target_length - current_length, sequence.shape[1]))
            return np.vstack([sequence, padding])
        elif current_length > target_length:
            return sequence[:target_length]
        
        return sequence
    
    def _empty_medication_features(self) -> Dict:
        """Return empty medication features"""
        return {
            'array': np.zeros(12),  # 10 categories + total + unique
            'names': ['antibiotic', 'anticoagulant', 'vasopressor', 'analgesic',
                     'sedative', 'insulin', 'steroid', 'antihypertensive',
                     'diuretic', 'beta_blocker', 'total_medications', 'unique_drugs'],
            'category_counts': {},
            'total_count': 0,
            'unique_count': 0
        }
    
    def _empty_outcomes(self) -> Dict:
        """Return empty outcomes"""
        from config import COMPLICATIONS
        
        outcome_names = sorted(COMPLICATIONS.keys())
        
        return {
            'array': np.zeros(len(outcome_names)),
            'names': outcome_names,
            'raw': {name: 0 for name in outcome_names}
        }
    
    def _generate_alignment_metadata(self, aligned_data: Dict) -> Dict:
        """Generate comprehensive metadata"""
        return {
            'time_series': {
                'sequence_length': aligned_data['time_series']['sequence_length'],
                'num_features': aligned_data['time_series']['num_features'],
                'preop_valid_steps': int(np.sum(aligned_data['attention_masks']['preoperative'])),
                'intraop_valid_steps': int(np.sum(aligned_data['attention_masks']['intraoperative']))
            },
            'text': {
                'preop_notes': aligned_data['text']['preoperative']['num_notes'],
                'intraop_notes': aligned_data['text']['intraoperative']['num_notes'],
                'embedding_dim': aligned_data['text']['combined_embedding'].shape[0]
            },
            'static': {
                'num_features': len(aligned_data['static']['array'])
            },
            'medications': {
                'total_count': aligned_data['medications'].get('total_count', 0)
            },
            'cross_modal': {
                'num_features': len(aligned_data['cross_modal_features']['array'])
            },
            'outcomes': {
                'num_complications': len(aligned_data['outcomes']['array'])
            }
        }
    
    def save_aligned_data(self, aligned_data: Dict, filepath: str):
        """Save aligned data to disk"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(aligned_data, f)
        
        print(f"Aligned data saved to: {filepath}")
    
    def load_aligned_data(self, filepath: str) -> Dict:
        """Load aligned data from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            aligned_data = pickle.load(f)
        
        print(f"Aligned data loaded from: {filepath}")
        return aligned_data