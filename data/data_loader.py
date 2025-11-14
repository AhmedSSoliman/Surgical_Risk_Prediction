# data/data_loader.py
"""
Data loader for MIMIC-III surgical patient cohort
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import MIMIC_PATH, COMPLICATIONS, TEMPORAL_WINDOWS

class MIMICDataLoader:
    """
    Load surgical patient data from MIMIC-III database
    
    Identifies surgical patients and loads:
    - Demographics
    - Procedures (surgical)
    - Clinical notes (with temporal classification)
    - Lab results
    - Vital signs
    - Medications
    - Outcomes (9 complications)
    """
    
    def __init__(self, mimic_path: str = None):
        self.mimic_path = Path(mimic_path or MIMIC_PATH)
        self._load_reference_tables()
        
    def _load_reference_tables(self):
        """Load reference tables"""
        print("Loading MIMIC-III reference tables...")
        
        try:
            self.d_icd_diagnoses = pd.read_csv(self.mimic_path / 'D_ICD_DIAGNOSES.csv')
            self.d_icd_procedures = pd.read_csv(self.mimic_path / 'D_ICD_PROCEDURES.csv')
            self.d_labitems = pd.read_csv(self.mimic_path / 'D_LABITEMS.csv')
            self.d_items = pd.read_csv(self.mimic_path / 'D_ITEMS.csv')
            print("✓ Reference tables loaded")
        except Exception as e:
            print(f"Error loading reference tables: {e}")
            raise
    
    def identify_surgical_cohort(self, 
                                n_patients: int = 1000,
                                major_surgery_only: bool = True) -> pd.DataFrame:
        """
        Identify surgical patient cohort
        
        Args:
            n_patients: Number of patients to load
            major_surgery_only: Filter to major surgeries only
            
        Returns:
            DataFrame with surgical admissions
        """
        print(f"Identifying surgical cohort (n={n_patients})...")
        
        # Load procedures
        procedures = pd.read_csv(self.mimic_path / 'PROCEDURES_ICD.csv')
        
        # Major surgery ICD-9 procedure codes (organ systems)
        major_surgery_prefixes = [
            '0',   # Nervous system
            '1',   # Endocrine system
            '2',   # Eye
            '3',   # Ear
            '4',   # Nose, mouth, pharynx
            '5',   # Respiratory system
            '6',   # Cardiovascular system
            '7',   # Hemic and lymphatic system
            '8',   # Digestive system
            '9'    # Urinary system
        ]
        
        if major_surgery_only:
            # Ensure ICD9_CODE is string type
            procedures['ICD9_CODE'] = procedures['ICD9_CODE'].astype(str)
            print(f"PROCEDURES_ICD.csv head:")
            print(procedures[['ICD9_CODE', 'HADM_ID']].head())
            print(f"Unique ICD9_CODE values: {procedures['ICD9_CODE'].unique()[:10]}")
            surgical_procedures = procedures[
                procedures['ICD9_CODE'].str[0].isin(major_surgery_prefixes)
            ]
            print(f"Filtered surgical_procedures: {len(surgical_procedures)}")
        else:
            surgical_procedures = procedures
        
        # Get unique admissions
        surgical_hadm_ids = surgical_procedures['HADM_ID'].unique()
        print(f"Unique surgical HADM_IDs: {len(surgical_hadm_ids)}")
        
        # Load admissions
        admissions = pd.read_csv(self.mimic_path / 'ADMISSIONS.csv')
        print(f"ADMISSIONS.csv head:")
        print(admissions[['HADM_ID']].head())
        surgical_admissions = admissions[admissions['HADM_ID'].isin(surgical_hadm_ids)]
        print(f"Admissions after filtering: {len(surgical_admissions)}")
        print(surgical_admissions.head())
        
        # Sample
        surgical_admissions = surgical_admissions.sample(
            min(n_patients, len(surgical_admissions)),
            random_state=42
        )
        
        print(f"✓ Identified {len(surgical_admissions)} surgical admissions")
        
        return surgical_admissions
    
    def load_patient_data(self, hadm_id: int) -> Dict:
        """
        Load complete data for a single patient admission
        
        Args:
            hadm_id: Hospital admission ID
            
        Returns:
            Dictionary with all patient data
        """
        print(f"Loading data for admission {hadm_id}...")
        
        patient_data = {
            'hadm_id': hadm_id,
            'admission': self._load_admission(hadm_id),
            'demographics': self._load_demographics(hadm_id),
            'procedures': self._load_procedures(hadm_id),
            'diagnoses': self._load_diagnoses(hadm_id),
            'notes': self._load_notes(hadm_id),
            'labs': self._load_labs(hadm_id),
            'vitals': self._load_vitals(hadm_id),
            'medications': self._load_medications(hadm_id),
            'outcomes': self._compute_outcomes(hadm_id),
            'surgery_time': self._estimate_surgery_time(hadm_id)
        }
        
        print(f"✓ Data loaded for admission {hadm_id}")
        
        return patient_data
    
    def _load_admission(self, hadm_id: int) -> Dict:
        """Load admission information"""
        admissions = pd.read_csv(self.mimic_path / 'ADMISSIONS.csv')
        admission = admissions[admissions['HADM_ID'] == hadm_id].iloc[0]
        return admission.to_dict()
    
    def _load_demographics(self, hadm_id: int) -> Dict:
        """Load patient demographics"""
        admissions = pd.read_csv(self.mimic_path / 'ADMISSIONS.csv')
        patients = pd.read_csv(self.mimic_path / 'PATIENTS.csv')
        
        admission = admissions[admissions['HADM_ID'] == hadm_id].iloc[0]
        patient = patients[patients['SUBJECT_ID'] == admission['SUBJECT_ID']].iloc[0]
        
        # Calculate age
        dob = pd.to_datetime(patient['DOB'])
        admit_time = pd.to_datetime(admission['ADMITTIME'])
        age = (admit_time - dob).days / 365.25
        
        return {
            'subject_id': int(admission['SUBJECT_ID']),
            'age': float(age),
            'gender': patient['GENDER'],
            'admission_type': admission['ADMISSION_TYPE'],
            'insurance': admission['INSURANCE'],
            'ethnicity': admission['ETHNICITY']
        }
    
    def _load_procedures(self, hadm_id: int) -> pd.DataFrame:
        """Load surgical procedures"""
        procedures = pd.read_csv(self.mimic_path / 'PROCEDURES_ICD.csv')
        procs = procedures[procedures['HADM_ID'] == hadm_id].copy()
        
        if not procs.empty:
            procs = procs.merge(
                self.d_icd_procedures[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']],
                on='ICD9_CODE',
                how='left'
            )
        
        return procs
    
    def _load_diagnoses(self, hadm_id: int) -> pd.DataFrame:
        """Load diagnoses"""
        diagnoses = pd.read_csv(self.mimic_path / 'DIAGNOSES_ICD.csv')
        diag = diagnoses[diagnoses['HADM_ID'] == hadm_id].copy()
        
        if not diag.empty:
            diag = diag.merge(
                self.d_icd_diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']],
                on='ICD9_CODE',
                how='left'
            )
        
        return diag
    
    def _load_notes(self, hadm_id: int) -> pd.DataFrame:
        """Load clinical notes"""
        noteevents = pd.read_csv(
            self.mimic_path / 'NOTEEVENTS.csv',
            low_memory=False
        )
        notes = noteevents[noteevents['HADM_ID'] == hadm_id].copy()
        
        if not notes.empty:
            notes['CHARTDATE'] = pd.to_datetime(notes['CHARTDATE'])
            notes = notes.sort_values('CHARTDATE')
        
        return notes[['CHARTDATE', 'CATEGORY', 'DESCRIPTION', 'TEXT']]
    
    def _load_labs(self, hadm_id: int) -> pd.DataFrame:
        """Load lab results"""
        # Load in chunks due to large file
        labevents = pd.read_csv(
            self.mimic_path / 'LABEVENTS.csv',
            chunksize=100000,
            low_memory=False
        )
        
        labs_list = []
        for chunk in labevents:
            labs_chunk = chunk[chunk['HADM_ID'] == hadm_id]
            if not labs_chunk.empty:
                labs_list.append(labs_chunk)
        
        if labs_list:
            labs = pd.concat(labs_list, ignore_index=True)
            labs = labs.merge(
                self.d_labitems[['ITEMID', 'LABEL', 'FLUID', 'CATEGORY']],
                on='ITEMID',
                how='left'
            )
            labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
            return labs[['CHARTTIME', 'ITEMID', 'LABEL', 'VALUENUM', 'VALUEUOM', 'FLAG']]
        
        return pd.DataFrame()
    
    def _load_vitals(self, hadm_id: int) -> pd.DataFrame:
        """Load vital signs"""
        # Load chartevents in chunks
        chartevents = pd.read_csv(
            self.mimic_path / 'CHARTEVENTS.csv',
            chunksize=1000000,
            low_memory=False
        )
        
        # Vital sign ITEMIDs
        from config import VITAL_ITEMS
        vital_itemids = [item for items in VITAL_ITEMS.values() for item in items]
        
        vitals_list = []
        for chunk in chartevents:
            vitals_chunk = chunk[
                (chunk['HADM_ID'] == hadm_id) &
                (chunk['ITEMID'].isin(vital_itemids))
            ]
            if not vitals_chunk.empty:
                vitals_list.append(vitals_chunk)
        
        if vitals_list:
            vitals = pd.concat(vitals_list, ignore_index=True)
            vitals = vitals.merge(
                self.d_items[['ITEMID', 'LABEL']],
                on='ITEMID',
                how='left'
            )
            vitals['CHARTTIME'] = pd.to_datetime(vitals['CHARTTIME'])
            return vitals[['CHARTTIME', 'ITEMID', 'LABEL', 'VALUENUM', 'VALUEUOM']]
        
        return pd.DataFrame()
    
    def _load_medications(self, hadm_id: int) -> pd.DataFrame:
        """Load medications"""
        prescriptions = pd.read_csv(self.mimic_path / 'PRESCRIPTIONS.csv')
        meds = prescriptions[prescriptions['HADM_ID'] == hadm_id].copy()
        
        if not meds.empty:
            meds['STARTDATE'] = pd.to_datetime(meds['STARTDATE'])
            meds = meds.sort_values('STARTDATE')
        
        return meds[['STARTDATE', 'DRUG', 'DRUG_TYPE', 'ROUTE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX']]
    
    def _estimate_surgery_time(self, hadm_id: int) -> pd.Timestamp:
        """
        Estimate surgery time from admission and notes
        
        Heuristics:
        1. Look for operative notes timestamp
        2. Use admission time + 24 hours as default
        """
        # Try to find operative note
        notes = self._load_notes(hadm_id)
        
        if not notes.empty:
            intraop_keywords = ['operative', 'or note', 'anesthesia', 'procedure']
            
            for _, note in notes.iterrows():
                category = str(note['CATEGORY']).lower()
                if any(kw in category for kw in intraop_keywords):
                    return note['CHARTDATE']
        
        # Default: admission time + 24 hours
        admission = self._load_admission(hadm_id)
        admit_time = pd.to_datetime(admission['ADMITTIME'])
        return admit_time + timedelta(hours=24)
    
    def _compute_outcomes(self, hadm_id: int) -> Dict:
        """
        Compute outcomes for 9 complications
        
        Returns:
            Dictionary with binary outcomes for each complication
        """
        outcomes = {}
        
        # Load necessary data
        admissions = pd.read_csv(self.mimic_path / 'ADMISSIONS.csv')
        admission = admissions[admissions['HADM_ID'] == hadm_id].iloc[0]
        
        diagnoses = self._load_diagnoses(hadm_id)
        
        try:
            icustays = pd.read_csv(self.mimic_path / 'ICUSTAYS.csv')
            icu_stays = icustays[icustays['HADM_ID'] == hadm_id]
        except:
            icu_stays = pd.DataFrame()
        
        # 1. Prolonged ICU stay
        if not icu_stays.empty:
            max_los_hours = (icu_stays['LOS'] * 24).max()
            outcomes['prolonged_icu'] = 1 if max_los_hours > 48 else 0
        else:
            outcomes['prolonged_icu'] = 0
        
        # 2. AKI
        outcomes['aki'] = self._check_icd_codes(diagnoses, COMPLICATIONS['aki']['icd9_codes'])
        
        # 3. Prolonged MV (approximate from procedures)
        outcomes['prolonged_mv'] = 0  # Would need ventilation data
        
        # 4. Wound complications
        outcomes['wound'] = self._check_icd_codes(diagnoses, COMPLICATIONS['wound']['icd9_codes'])
        
        # 5. Neurological
        outcomes['neurological'] = self._check_icd_codes(diagnoses, COMPLICATIONS['neurological']['icd9_codes'])
        
        # 6. Sepsis
        outcomes['sepsis'] = self._check_icd_codes(diagnoses, COMPLICATIONS['sepsis']['icd9_codes'])
        
        # 7. Cardiovascular
        outcomes['cardiovascular'] = self._check_icd_codes(diagnoses, COMPLICATIONS['cardiovascular']['icd9_codes'])
        
        # 8. VTE
        outcomes['vte'] = self._check_icd_codes(diagnoses, COMPLICATIONS['vte']['icd9_codes'])
        
        # 9. Mortality
        outcomes['mortality'] = 1 if pd.notna(admission['DEATHTIME']) else 0
        
        return outcomes
    
    def _check_icd_codes(self, diagnoses: pd.DataFrame, code_list: List[str]) -> int:
        """Check if any diagnosis matches ICD code list"""
        if diagnoses.empty:
            return 0
        
        for code in code_list:
            if diagnoses['ICD9_CODE'].astype(str).str.startswith(code).any():
                return 1
        
        return 0


class SampleDataGenerator:
    """Generate realistic sample data for testing"""
    
    @staticmethod
    def generate_sample_patient() -> Dict:
        """Generate complete sample patient data"""
        
        hadm_id = np.random.randint(100000, 999999)
        surgery_time = pd.Timestamp.now() - timedelta(days=2)
        
        # Generate sample data (implementation from previous version)
        # ... (sample generation code similar to before)
        
        return {
            'hadm_id': hadm_id,
            'surgery_time': surgery_time,
            # ... other sample data
        }