# preprocessing/preprocess_time_series.py
"""
Advanced Time Series Preprocessing for Labs and Vitals
Handles irregular sampling, missing data, outliers, and feature extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import interpolate, signal, stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings('ignore')

from config import TIME_SERIES_CONFIG, LAB_ITEMS, VITAL_ITEMS, TEMPORAL_WINDOWS

class TimeSeriesPreprocessor:
    """
    Advanced preprocessor for time series data (labs and vitals)
    
    Features:
    - Preoperative and intraoperative filtering
    - Temporal alignment to surgery time
    - Outlier detection and removal
    - Missing data imputation (multiple methods)
    - Feature extraction (statistical + temporal)
    - Normalization
    - Sequence padding/truncation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or TIME_SERIES_CONFIG
        self.scaler_type = self.config.get('normalization', 'robust')
        
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.feature_names = []
        self.fitted = False
        
    def preprocess_labs(self, 
                       labs_df: pd.DataFrame, 
                       surgery_time: pd.Timestamp,
                       phase: str = 'preoperative') -> Tuple[np.ndarray, List[str], Dict]:
        """
        Preprocess laboratory results for specific surgical phase
        
        Args:
            labs_df: DataFrame with columns [CHARTTIME, ITEMID, VALUENUM]
            surgery_time: Surgery start time
            phase: 'preoperative' or 'intraoperative'
            
        Returns:
            processed_array: (sequence_length, num_features)
            feature_names: List of feature names
            metadata: Processing metadata
        """
        if labs_df.empty or surgery_time is None:
            return self._empty_result(len(LAB_ITEMS), 'labs')
        
        # Filter by phase
        labs_df = self._filter_by_phase(labs_df, surgery_time, phase, 'labs')
        
        if labs_df.empty:
            return self._empty_result(len(LAB_ITEMS), 'labs')
        
        # Create time series for each lab
        lab_series = {}
        lab_metadata = {}
        
        # Remove duplicate columns and index before processing
        labs_df = labs_df.loc[:, ~labs_df.columns.duplicated()]
        labs_df = labs_df[~labs_df.index.duplicated()]
        for lab_name, itemids in LAB_ITEMS.items():
            lab_data = labs_df[labs_df['ITEMID'].isin(itemids)].copy()
            # Remove duplicates in lab_data as well
            lab_data = lab_data.loc[:, ~lab_data.columns.duplicated()]
            lab_data = lab_data[~lab_data.index.duplicated()]
            if not lab_data.empty:
                # Remove outliers
                original_count = len(lab_data)
                lab_data = self._remove_outliers(lab_data, 'VALUENUM')
                outliers_removed = original_count - len(lab_data)
                
                # Sort by time
                lab_data = lab_data.sort_values('CHARTTIME')
                
                # Calculate statistics before alignment
                lab_metadata[lab_name] = {
                    'count': len(lab_data),
                    'mean': float(lab_data['VALUENUM'].mean()),
                    'std': float(lab_data['VALUENUM'].std()),
                    'min': float(lab_data['VALUENUM'].min()),
                    'max': float(lab_data['VALUENUM'].max()),
                    'outliers_removed': outliers_removed
                }
                
                # Create time series
                lab_series[lab_name] = lab_data[['CHARTTIME', 'VALUENUM']].set_index('CHARTTIME')
        
        # Determine time window
        try:
            if phase == 'preoperative':
                window_start = surgery_time - TEMPORAL_WINDOWS['preoperative']['labs']
                window_end = surgery_time
            else:  # intraoperative
                window_start = surgery_time
                window_end = surgery_time + TEMPORAL_WINDOWS['intraoperative']['duration']
        except (OverflowError, ValueError):
            # If timestamp calculation fails, return empty result
            return self._empty_result(len(LAB_ITEMS), 'labs')
        
        # Align to common timeline
        aligned_series = self._align_to_timeline(
            lab_series, 
            window_start, 
            window_end,
            self.config['sampling_rate']
        )
        
        # Check if we have any features after alignment
        if aligned_series.shape[1] == 0:
            return self._empty_result(len(LAB_ITEMS), 'labs')
        
        # Impute missing values
        imputed_series = self._impute_missing(
            aligned_series, 
            method=self.config['imputation_method']
        )
        
        # Extract statistical features
        statistical_features = self._extract_statistical_features(imputed_series)
        
        # Normalize
        normalized_series = self._normalize(imputed_series)
        
        # Ensure correct shape
        target_length = self.config['max_sequence_length']
        normalized_series = self._pad_or_truncate(normalized_series, target_length)
        
        # Feature names
        feature_names = list(LAB_ITEMS.keys())
        
        # Metadata
        metadata = {
            'phase': phase,
            'window_start': str(window_start),
            'window_end': str(window_end),
            'original_length': aligned_series.shape[0],
            'final_length': normalized_series.shape[0],
            'num_features': normalized_series.shape[1],
            'lab_metadata': lab_metadata,
            'statistical_features': statistical_features
        }
        
        return normalized_series, feature_names, metadata
    
    def preprocess_vitals(self,
                         vitals_df: pd.DataFrame,
                         surgery_time: pd.Timestamp,
                         phase: str = 'preoperative') -> Tuple[np.ndarray, List[str], Dict]:
        """
        Preprocess vital signs for specific surgical phase
        
        Args:
            vitals_df: DataFrame with columns [CHARTTIME, ITEMID, VALUENUM]
            surgery_time: Surgery start time
            phase: 'preoperative' or 'intraoperative'
            
        Returns:
            processed_array: (sequence_length, num_features)
            feature_names: List of feature names
            metadata: Processing metadata
        """
        if vitals_df.empty or surgery_time is None:
            return self._empty_result(len(VITAL_ITEMS) + 3, 'vitals')
        
        # Filter by phase
        vitals_df = self._filter_by_phase(vitals_df, surgery_time, phase, 'vitals')
        
        if vitals_df.empty:
            return self._empty_result(len(VITAL_ITEMS) + 3, 'vitals')
        
        # Create time series for each vital sign
        vital_series = {}
        vital_metadata = {}
        
        for vital_name, itemids in VITAL_ITEMS.items():
            vital_data = vitals_df[vitals_df['ITEMID'].isin(itemids)].copy()
            
            if not vital_data.empty:
                # Remove outliers
                original_count = len(vital_data)
                vital_data = self._remove_outliers(vital_data, 'VALUENUM')
                outliers_removed = original_count - len(vital_data)
                
                # Special handling for blood pressure (take mean if multiple measurements)
                if vital_name in ['SBP', 'DBP', 'MBP']:
                    vital_data = vital_data.groupby('CHARTTIME')['VALUENUM'].mean().reset_index()
                
                # Sort by time
                vital_data = vital_data.sort_values('CHARTTIME')
                
                # Calculate statistics
                vital_metadata[vital_name] = {
                    'count': len(vital_data),
                    'mean': float(vital_data['VALUENUM'].mean()),
                    'std': float(vital_data['VALUENUM'].std()),
                    'min': float(vital_data['VALUENUM'].min()),
                    'max': float(vital_data['VALUENUM'].max()),
                    'outliers_removed': outliers_removed
                }
                
                # Create time series
                vital_series[vital_name] = vital_data.set_index('CHARTTIME')['VALUENUM']
        
        # Determine time window
        try:
            if phase == 'preoperative':
                window_start = surgery_time - TEMPORAL_WINDOWS['preoperative']['vitals']
                window_end = surgery_time
            else:  # intraoperative
                window_start = surgery_time
                window_end = surgery_time + TEMPORAL_WINDOWS['intraoperative']['vitals']
        except (OverflowError, ValueError):
            # If timestamp calculation fails, return empty result
            return self._empty_result(len(VITAL_ITEMS), 'vitals')
        
        # Align to common timeline
        aligned_series = self._align_to_timeline(
            vital_series,
            window_start,
            window_end,
            self.config['sampling_rate']
        )
        
        # Check if we have any features after alignment
        if aligned_series.shape[1] == 0:
            return self._empty_result(len(VITAL_ITEMS), 'vitals')
        
        # Impute missing values (use linear interpolation for vitals)
        imputed_series = self._impute_missing(aligned_series, method='linear')
        
        # Calculate derived vital features
        derived_features = self._calculate_derived_vitals(imputed_series)
        
        # Combine original and derived
        combined_series = np.hstack([imputed_series, derived_features])
        
        # Normalize
        normalized_series = self._normalize(combined_series)
        
        # Ensure correct shape
        target_length = self.config['max_sequence_length']
        normalized_series = self._pad_or_truncate(normalized_series, target_length)
        
        # Feature names
        feature_names = list(VITAL_ITEMS.keys()) + ['Shock_Index', 'MAP_Calculated', 'Pulse_Pressure']
        
        # Extract statistical features
        statistical_features = self._extract_statistical_features(normalized_series)
        
        # Metadata
        metadata = {
            'phase': phase,
            'window_start': str(window_start),
            'window_end': str(window_end),
            'original_length': aligned_series.shape[0],
            'final_length': normalized_series.shape[0],
            'num_features': normalized_series.shape[1],
            'vital_metadata': vital_metadata,
            'statistical_features': statistical_features
        }
        
        return normalized_series, feature_names, metadata
    
    def _filter_by_phase(self, 
                        df: pd.DataFrame, 
                        surgery_time: pd.Timestamp,
                        phase: str,
                        data_type: str) -> pd.DataFrame:
        """Filter data by surgical phase"""
        df = df.copy()
        df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'])
        
        try:
            if phase == 'preoperative':
                window = TEMPORAL_WINDOWS['preoperative'][data_type]
                start_time = surgery_time - window
                end_time = surgery_time
            elif phase == 'intraoperative':
                start_time = surgery_time
                if data_type == 'vitals':
                    end_time = surgery_time + TEMPORAL_WINDOWS['intraoperative']['vitals']
                else:
                    end_time = surgery_time + TEMPORAL_WINDOWS['intraoperative']['duration']
            else:
                raise ValueError(f"Unknown phase: {phase}")
        except (OverflowError, ValueError) as e:
            if "Unknown phase" in str(e):
                raise
            # If timestamp calculation fails, return empty DataFrame
            return pd.DataFrame()
        
        # Filter data
        filtered_df = df[
            (df['CHARTTIME'] >= start_time) & 
            (df['CHARTTIME'] <= end_time)
        ]
        
        return filtered_df
    
    def _remove_outliers(self, 
                        df: pd.DataFrame, 
                        column: str, 
                        method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers using specified method
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' or 'zscore'
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            threshold = self.config['outlier_threshold']
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column]))
            return df[z_scores < self.config['outlier_threshold']]
        
        return df
    
    def _align_to_timeline(self,
                          series_dict: Dict[str, pd.Series],
                          start_time: pd.Timestamp,
                          end_time: pd.Timestamp,
                          freq: str) -> np.ndarray:
        """Align multiple time series to common timeline"""
        # Create common timeline
        timeline = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Align each series
        aligned_data = []
        for series_name in sorted(series_dict.keys()):
            series = series_dict[series_name]
            
            # Remove duplicate index labels before reindexing
            if isinstance(series, pd.DataFrame):
                series = series.loc[:, ~series.columns.duplicated()]
            series = series[~series.index.duplicated()]
            
            # Reindex to common timeline
            if isinstance(series, pd.DataFrame):
                reindexed = series.reindex(timeline).values.flatten()
            else:
                reindexed = series.reindex(timeline).values
            
            aligned_data.append(reindexed)
        
        # Stack into matrix
        if aligned_data:
            return np.column_stack(aligned_data)
        else:
            return np.zeros((len(timeline), len(series_dict)))
    
    def _impute_missing(self, 
                       data: np.ndarray, 
                       method: str = 'forward_fill') -> np.ndarray:
        """Impute missing values using specified method"""
        df = pd.DataFrame(data)
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'linear':
            df = df.interpolate(method='linear', limit_direction='both')
        elif method == 'spline':
            df = df.interpolate(method='spline', order=2, limit_direction='both')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        elif method == 'knn':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        return df.values
    
    def _calculate_derived_vitals(self, vitals_array: np.ndarray) -> np.ndarray:
        """
        Calculate derived vital sign features
        
        Order assumed: HR, SBP, DBP, MBP, RR, Temp, SpO2, FiO2
        """
        derived = []
        
        num_features = vitals_array.shape[1]
        
        # Shock Index (HR / SBP)
        if num_features >= 2:
            hr = vitals_array[:, 0]
            sbp = vitals_array[:, 1]
            shock_index = hr / (sbp + 1e-8)
            derived.append(shock_index)
        else:
            derived.append(np.zeros(vitals_array.shape[0]))
        
        # Mean Arterial Pressure (calculated from SBP and DBP)
        if num_features >= 3:
            sbp = vitals_array[:, 1]
            dbp = vitals_array[:, 2]
            map_calc = (sbp + 2 * dbp) / 3
            derived.append(map_calc)
        else:
            derived.append(np.zeros(vitals_array.shape[0]))
        
        # Pulse Pressure (SBP - DBP)
        if num_features >= 3:
            sbp = vitals_array[:, 1]
            dbp = vitals_array[:, 2]
            pp = sbp - dbp
            derived.append(pp)
        else:
            derived.append(np.zeros(vitals_array.shape[0]))
        
        # SpO2/FiO2 ratio (if both available)
        if num_features >= 8:
            spo2 = vitals_array[:, 6]
            fio2 = vitals_array[:, 7]
            sf_ratio = spo2 / (fio2 + 1e-8)
            derived.append(sf_ratio)
        
        if derived:
            return np.column_stack(derived)
        else:
            return np.zeros((vitals_array.shape[0], 3))
    
    def _extract_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive statistical features"""
        features = {}
        
        for i in range(data.shape[1]):
            series = data[:, i]
            
            # Basic statistics
            features[f'mean_{i}'] = float(np.mean(series))
            features[f'std_{i}'] = float(np.std(series))
            features[f'min_{i}'] = float(np.min(series))
            features[f'max_{i}'] = float(np.max(series))
            features[f'median_{i}'] = float(np.median(series))
            features[f'q25_{i}'] = float(np.percentile(series, 25))
            features[f'q75_{i}'] = float(np.percentile(series, 75))
            features[f'range_{i}'] = float(np.max(series) - np.min(series))
            features[f'iqr_{i}'] = float(np.percentile(series, 75) - np.percentile(series, 25))
            
            # Coefficient of variation
            mean_val = np.mean(series)
            features[f'cv_{i}'] = float(np.std(series) / (mean_val + 1e-8))
            
            # Trend (slope)
            features[f'slope_{i}'] = self._calculate_slope(series)
            
            # Variability measures
            features[f'variance_{i}'] = float(np.var(series))
            features[f'skewness_{i}'] = float(stats.skew(series))
            features[f'kurtosis_{i}'] = float(stats.kurtosis(series))
            
            # Rate of change
            if len(series) > 1:
                diffs = np.diff(series)
                features[f'mean_diff_{i}'] = float(np.mean(diffs))
                features[f'std_diff_{i}'] = float(np.std(diffs))
                features[f'max_diff_{i}'] = float(np.max(np.abs(diffs)))
        
        return features
    
    def _calculate_slope(self, series: np.ndarray) -> float:
        """Calculate linear trend slope using least squares"""
        x = np.arange(len(series))
        if len(series) > 1:
            coeffs = np.polyfit(x, series, 1)
            return float(coeffs[0])
        return 0.0
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using configured scaler, refitting if feature count changes"""
        if data.shape[0] == 0 or data.shape[1] == 0:
            return data
        
        # Track feature count for scaler
        if not hasattr(self, '_scaler_feature_count'):
            self._scaler_feature_count = None
        # If not fitted or feature count changed, refit
        if not self.fitted or self._scaler_feature_count != data.shape[1]:
            normalized = self.scaler.fit_transform(data)
            self.fitted = True
            self._scaler_feature_count = data.shape[1]
        else:
            normalized = self.scaler.transform(data)
        return normalized
    
    def _pad_or_truncate(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate sequence to target length"""
        current_length = data.shape[0]
        
        if current_length < target_length:
            # Pad with zeros
            padding = np.zeros((target_length - current_length, data.shape[1]))
            return np.vstack([data, padding])
        elif current_length > target_length:
            # Truncate (keep most recent)
            return data[-target_length:]
        else:
            return data
    
    def _empty_result(self, num_features: int, data_type: str) -> Tuple[np.ndarray, List[str], Dict]:
        """Return empty result with correct shape"""
        target_length = self.config['max_sequence_length']
        
        if data_type == 'vitals':
            num_features = num_features + 3  # Add derived features
        
        empty_array = np.zeros((target_length, num_features))
        
        if data_type == 'labs':
            feature_names = list(LAB_ITEMS.keys())
        else:
            feature_names = list(VITAL_ITEMS.keys()) + ['Shock_Index', 'MAP_Calculated', 'Pulse_Pressure']
        
        metadata = {
            'phase': 'unknown',
            'window_start': 'N/A',
            'window_end': 'N/A',
            'original_length': 0,
            'final_length': target_length,
            'num_features': num_features,
            f'{data_type}_metadata': {},
            'statistical_features': {}
        }
        
        return empty_array, feature_names, metadata
    
    def extract_tsfresh_features(self, 
                                time_series_df: pd.DataFrame,
                                id_column: str = 'id',
                                time_column: str = 'time',
                                value_column: str = 'value') -> pd.DataFrame:
        """
        Extract advanced time series features using tsfresh
        
        Args:
            time_series_df: DataFrame in tsfresh format
            id_column: Column name for series ID
            time_column: Column name for time
            value_column: Column name for values
            
        Returns:
            DataFrame with extracted features
        """
        from tsfresh import extract_features
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        
        # Extract features
        features = extract_features(
            time_series_df,
            column_id=id_column,
            column_sort=time_column,
            default_fc_parameters=ComprehensiveFCParameters(),
            impute_function=impute
        )
        
        return features