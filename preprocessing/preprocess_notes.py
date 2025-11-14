# preprocessing/preprocess_notes.py
"""
Clinical Notes Preprocessing with Preoperative/Intraoperative Filtering
Handles medical NLP, temporal classification, and embeddings
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, NOTE_CATEGORIES, TEMPORAL_WINDOWS

class ClinicalNotesPreprocessor:
    """
    Advanced preprocessor for clinical notes with temporal awareness
    
    Features:
    - Preoperative/intraoperative/postoperative note filtering
    - Medical entity recognition
    - Contextualized embeddings (BioBERT/PubMedBERT)
    - Complication mention extraction
    - Temporal ordering
    - Section-based processing
    """
    
    def __init__(self, model_name: str = None, use_transformer: bool = False):
        """
        Initialize notes preprocessor
        
        Args:
            model_name: Transformer model name (default: PubMedBERT)
            use_transformer: Whether to use transformer embeddings (disabled by default due to MPS stability issues)
        """
        self.model_name = model_name or MODEL_CONFIG['vibe_tuning']['base_model']
        self.max_length = MODEL_CONFIG['vibe_tuning']['max_length']
        self.use_transformer = use_transformer
        
        # Load spaCy model (medical if available)
        try:
            self.nlp = spacy.load("en_core_sci_md")
            print("Loaded medical spaCy model: en_core_sci_md")
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Loaded standard spaCy model: en_core_web_sm")
            except:
                print("No spaCy model found. Installing...")
                import os
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        
        # Setup device - prioritize MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("✓ Apple Silicon GPU (MPS) detected and will be used")
            # Clear MPS cache at the start
            torch.mps.empty_cache()
            print("✓ MPS cache cleared")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("✓ NVIDIA GPU (CUDA) detected and will be used")
            # Clear CUDA cache at the start
            torch.cuda.empty_cache()
            print("✓ CUDA cache cleared")
        else:
            self.device = torch.device('cpu')
            print("Using CPU (no GPU detected)")
        
        # Clear Python garbage collector
        import gc
        gc.collect()
        
        # Load transformer model with MPS-optimized settings
        self.model = None
        self.tokenizer = None
        
        if self.use_transformer:
            print(f"Loading transformer model: {self.model_name}")
            try:
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    model_max_length=512
                )
                print("✓ Tokenizer loaded")
                
                # Load model weights on CPU first with explicit dtype
                print("Loading model weights on CPU...")
                import gc
                gc.collect()
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                print("✓ Model loaded on CPU")
                
                # Move to MPS if available, with error handling
                if self.device.type == 'mps':
                    try:
                        print("Moving model to MPS (this may take a moment)...")
                        # Move in eval mode to reduce memory
                        self.model = self.model.to(self.device)
                        
                        # Warm up with a dummy input
                        print("Warming up MPS kernels...")
                        dummy_input = self.tokenizer(
                            "test", 
                            return_tensors='pt',
                            max_length=128,  # Shorter for warmup
                            padding='max_length',
                            truncation=True
                        )
                        dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                        with torch.no_grad():
                            _ = self.model(**dummy_input)
                        del dummy_input
                        torch.mps.empty_cache()
                        gc.collect()
                        print(f"✓ Model successfully loaded on {self.device}")
                        
                    except Exception as mps_error:
                        print(f"WARNING: MPS loading failed: {mps_error}")
                        print("Falling back to CPU for transformer model")
                        self.device = torch.device('cpu')
                        self.model = self.model.to(self.device)
                        print(f"✓ Model loaded on CPU")
                else:
                    # Already on CPU or CUDA
                    self.model = self.model.to(self.device)
                    print(f"✓ Model loaded on {self.device}")
                
            except Exception as e:
                print(f"ERROR loading transformer model: {type(e).__name__}: {e}")
                print("Falling back to simple bag-of-words embeddings")
                self.model = None
                self.tokenizer = None
                self.device = torch.device('cpu')
        else:
            print("Using simple bag-of-words embeddings (transformer disabled)")
        
        # Medical patterns
        self.medical_patterns = self._compile_medical_patterns()
        
        # Intraoperative note keywords
        self.intraop_keywords = TEMPORAL_WINDOWS['intraoperative']['notes']
        
    def preprocess_notes(self,
                        notes_df: pd.DataFrame,
                        surgery_time: pd.Timestamp = None,
                        phase: str = 'all') -> Dict:
        """
        Preprocess clinical notes with temporal filtering
        
        Args:
            notes_df: DataFrame with [CHARTDATE, CATEGORY, TEXT]
            surgery_time: Surgery start timestamp
            phase: 'preoperative', 'intraoperative', 'postoperative', or 'all'
            
        Returns:
            Dictionary with processed notes and features
        """
        if notes_df.empty:
            return self._empty_result()
        
        # Filter notes by phase
        if phase != 'all' and surgery_time is not None:
            notes_df = self._filter_notes_by_phase(notes_df, surgery_time, phase)
        
        if notes_df.empty:
            return self._empty_result()
        
        # Sort by time
        if 'CHARTDATE' in notes_df.columns:
            notes_df = notes_df.copy()
            notes_df['CHARTDATE'] = pd.to_datetime(notes_df['CHARTDATE'])
            notes_df = notes_df.sort_values('CHARTDATE')
        
        processed = {
            'raw_texts': [],
            'cleaned_texts': [],
            'embeddings': [],
            'entities': [],
            'categories': [],
            'temporal_phases': [],
            'temporal_positions': [],
            'sentences': [],
            'section_embeddings': {},
            'complication_mentions': {},
            'severity_indicators': {},
            'metadata': {}
        }
        
        # Process each note
        for idx, row in notes_df.iterrows():
            text = str(row['TEXT'])
            category = row.get('CATEGORY', 'Unknown')
            chart_time = row.get('CHARTDATE')
            
            # Clean text
            cleaned = self._clean_text(text)
            
            # Skip if too short
            if len(cleaned.split()) < 10:
                continue
            
            # Determine temporal phase
            temporal_phase = self._determine_temporal_phase(
                category, cleaned, chart_time, surgery_time
            )
            
            # Extract entities
            entities = self._extract_entities(cleaned)
            
            # Generate embeddings
            embedding = self._generate_embedding(cleaned)
            
            # Segment into sentences
            sentences = self._segment_sentences(cleaned)
            
            # Calculate temporal position (hours from surgery)
            if surgery_time and chart_time:
                try:
                    # Validate timestamps to prevent overflow
                    if pd.notnull(chart_time) and pd.notnull(surgery_time):
                        # Calculate difference safely
                        time_diff_seconds = (chart_time - surgery_time).total_seconds()
                        # Clamp to reasonable range (-30 days to +30 days)
                        max_seconds = 30 * 24 * 3600
                        if abs(time_diff_seconds) > max_seconds:
                            temporal_pos = 0  # Use neutral value for unreasonable times
                        else:
                            temporal_pos = time_diff_seconds / 3600
                    else:
                        temporal_pos = 0
                except (OverflowError, ValueError, AttributeError):
                    # Handle any timestamp calculation errors
                    temporal_pos = 0
            else:
                temporal_pos = 0
            
            # Store processed data
            processed['raw_texts'].append(text)
            processed['cleaned_texts'].append(cleaned)
            processed['embeddings'].append(embedding)
            processed['entities'].append(entities)
            processed['categories'].append(category)
            processed['temporal_phases'].append(temporal_phase)
            processed['temporal_positions'].append(temporal_pos)
            processed['sentences'].append(sentences)
        
        # Extract section-specific embeddings
        processed['section_embeddings'] = self._extract_section_embeddings(notes_df)
        
        # Extract complication mentions
        all_text = ' '.join(processed['cleaned_texts'])
        processed['complication_mentions'] = self._extract_complication_mentions(all_text)
        
        # Extract severity indicators
        processed['severity_indicators'] = self._extract_severity_indicators(all_text)
        
        # Generate aggregated embeddings by phase
        processed['aggregated_embeddings'] = self._aggregate_embeddings_by_phase(
            processed['embeddings'],
            processed['temporal_phases']
        )
        
        # Generate metadata
        processed['metadata'] = self._generate_metadata(processed, phase)
        
        return processed
    
    def _filter_notes_by_phase(self,
                               notes_df: pd.DataFrame,
                               surgery_time: pd.Timestamp,
                               phase: str) -> pd.DataFrame:
        """
        Filter notes by surgical phase
        
        Args:
            notes_df: Notes DataFrame
            surgery_time: Surgery timestamp
            phase: 'preoperative', 'intraoperative', or 'postoperative'
        """
        filtered_notes = []
        
        for idx, row in notes_df.iterrows():
            chart_time = pd.to_datetime(row.get('CHARTDATE'))
            category = row.get('CATEGORY', '')
            text = str(row.get('TEXT', ''))
            
            note_phase = self._determine_temporal_phase(
                category, text, chart_time, surgery_time
            )
            
            if note_phase == phase:
                filtered_notes.append(row)
        
        if filtered_notes:
            return pd.DataFrame(filtered_notes)
        else:
            return pd.DataFrame()
    
    def _determine_temporal_phase(self,
                                  category: str,
                                  text: str,
                                  chart_time: pd.Timestamp,
                                  surgery_time: pd.Timestamp) -> str:
        """
        Determine temporal phase of clinical note
        
        Priority:
        1. Keyword-based detection (for intraoperative)
        2. Category-based classification
        3. Time-based classification
        """
        # Check for intraoperative keywords
        text_lower = text.lower()
        category_lower = category.lower()
        
        # Intraoperative detection
        intraop_keywords = ['operative note', 'or note', 'anesthesia', 'intraoperative',
                           'procedure note', 'surgery note', 'operative report']
        
        if any(keyword in category_lower for keyword in self.intraop_keywords):
            return 'intraoperative'
        
        if any(keyword in text_lower[:500] for keyword in intraop_keywords):
            return 'intraoperative'
        
        # Category-based classification
        if category in NOTE_CATEGORIES.get('preoperative', []):
            return 'preoperative'
        elif category in NOTE_CATEGORIES.get('intraoperative', []):
            return 'intraoperative'
        elif category in NOTE_CATEGORIES.get('postoperative', []):
            return 'postoperative'
        
        # Time-based classification
        if chart_time and surgery_time:
            try:
                # Validate timestamps
                if pd.notnull(chart_time) and pd.notnull(surgery_time):
                    time_diff = chart_time - surgery_time
                    
                    if time_diff < timedelta(0):  # Before surgery
                        # Check if within preoperative window
                        if abs(time_diff) <= TEMPORAL_WINDOWS['preoperative']['notes']:
                            return 'preoperative'
                    elif time_diff <= TEMPORAL_WINDOWS['intraoperative']['duration']:
                        return 'intraoperative'
                    else:
                        return 'postoperative'
            except (OverflowError, ValueError, AttributeError):
                # If timestamp calculation fails, return unknown
                pass
        
        return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize clinical text"""
        # Remove PHI markers (MIMIC-III format)
        text = re.sub(r'\[\*\*[^\]]+\*\*\]', '[REDACTED]', text)
        
        # Remove dates and times (but keep temporal markers)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '[DATE]', text)
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?', '[TIME]', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\.\,\-\/\(\)\%\:\;\[\]]', '', text)
        
        # Normalize case
        text = text.lower()
        
        # Remove very short tokens
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 1]
        
        return ' '.join(tokens)
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using spaCy"""
        # Limit text length for performance
        text = text[:100000]
        
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate contextualized embedding using transformer model
        Falls back to simple TF-IDF if model unavailable
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Fallback to simple embeddings if model failed to load
        if self.model is None or self.tokenizer is None:
            return self._generate_simple_embedding(text)
        
        try:
            # Tokenize with truncation
            inputs = self.tokenizer(
                text,
                max_length=min(512, self.max_length),  # Limit to 512 for safety
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"WARNING: Embedding generation failed: {e}. Using fallback.")
            return self._generate_simple_embedding(text)
    
    def _generate_simple_embedding(self, text: str, dim: int = 768) -> np.ndarray:
        """
        Generate simple bag-of-words embedding as fallback
        
        Args:
            text: Input text
            dim: Embedding dimension (default 768 to match BERT)
            
        Returns:
            Simple embedding vector
        """
        # Simple word count based features
        words = text.lower().split()
        
        # Create a deterministic hash-based embedding
        embedding = np.zeros(dim)
        for i, word in enumerate(words[:100]):  # Limit to first 100 words
            # Use hash to create consistent embedding
            idx = hash(word) % dim
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        doc = self.nlp(text[:100000])  # Limit for performance
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences
    
    def _extract_section_embeddings(self, notes_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for specific clinical note sections
        
        Common sections:
        - Chief Complaint
        - History of Present Illness
        - Past Medical History
        - Physical Examination
        - Assessment
        - Plan
        """
        section_headers = {
            'chief_complaint': ['chief complaint', 'cc:', 'presenting complaint'],
            'hpi': ['history of present illness', 'hpi:', 'present illness'],
            'pmh': ['past medical history', 'pmh:', 'medical history'],
            'physical_exam': ['physical examination', 'physical exam', 'exam:', 'pe:'],
            'assessment': ['assessment:', 'impression:', 'diagnosis:'],
            'plan': ['plan:', 'recommendation:', 'disposition:'],
            'complications': ['complications:', 'adverse events:', 'postoperative complications']
        }
        
        section_embeddings = {}
        all_text = ' '.join(notes_df['TEXT'].astype(str))
        
        for section_name, keywords in section_headers.items():
            section_text = self._extract_section_text(all_text, keywords)
            
            if section_text and len(section_text.split()) > 5:
                embedding = self._generate_embedding(section_text)
                section_embeddings[section_name] = embedding
        
        return section_embeddings
    
    def _extract_section_text(self, text: str, keywords: List[str]) -> str:
        """Extract text from specific section"""
        text_lower = text.lower()
        
        for keyword in keywords:
            # Find section start
            pattern = rf'{keyword}\s*(.*?)(?=\n\s*[a-z\s]+:|$)'
            matches = re.findall(pattern, text_lower, re.DOTALL)
            
            if matches:
                # Return first substantial match
                for match in matches:
                    if len(match.split()) > 5:
                        return match.strip()
        
        return ''
    
    def _compile_medical_patterns(self) -> Dict[str, List[str]]:
        """Compile comprehensive medical regex patterns"""
        return {
            'complications': [
                r'complication[s]?',
                r'adverse event[s]?',
                r'postoperative infection',
                r'wound infection',
                r'surgical site infection|ssi',
                r'sepsis',
                r'septic shock',
                r'aki|acute kidney injury',
                r'acute renal failure|arf',
                r'renal failure',
                r'stroke|cva|cerebrovascular accident',
                r'delirium',
                r'confusion',
                r'altered mental status',
                r'arrhythmia',
                r'atrial fibrillation|afib',
                r'myocardial infarction|mi',
                r'cardiac arrest',
                r'pulmonary embolism|pe',
                r'deep vein thrombosis|dvt',
                r'thromboembolism',
                r'respiratory failure',
                r'pneumonia',
                r'ards',
                r'wound dehiscence',
                r'anastomotic leak',
                r'bleeding|hemorrhage'
            ],
            'severity': [
                r'critical',
                r'severe',
                r'serious',
                r'life-threatening',
                r'emergency',
                r'urgent',
                r'emergent',
                r'rapidly',
                r'acute',
                r'significant'
            ],
            'improvement': [
                r'improving',
                r'better',
                r'stable',
                r'resolved',
                r'recovery',
                r'improving',
                r'normalized',
                r'afebrile',
                r'ambulat'
            ],
            'deterioration': [
                r'worsening',
                r'deteriorating',
                r'declining',
                r'unstable',
                r'decompensating',
                r'failure',
                r'requiring',
                r'urgent'
            ],
            'monitoring': [
                r'monitor',
                r'observe',
                r'watch',
                r'follow',
                r'serial',
                r'frequent'
            ]
        }
    
    def _extract_complication_mentions(self, text: str) -> Dict[str, Dict]:
        """Extract and count complication mentions with context"""
        mentions = {}
        text_lower = text.lower()
        
        for pattern in self.medical_patterns['complications']:
            matches = list(re.finditer(pattern, text_lower))
            
            if matches:
                contexts = []
                for match in matches:
                    # Get surrounding context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text_lower), match.end() + 50)
                    context = text_lower[start:end]
                    contexts.append(context)
                
                mentions[pattern] = {
                    'count': len(matches),
                    'contexts': contexts[:3]  # Store first 3 contexts
                }
        
        return mentions
    
    def _extract_severity_indicators(self, text: str) -> Dict[str, int]:
        """Extract severity indicators from text"""
        indicators = {
            'critical': 0,
            'severe': 0,
            'moderate': 0,
            'mild': 0
        }
        
        text_lower = text.lower()
        
        # Critical
        for pattern in self.medical_patterns['severity']:
            indicators['critical'] += len(re.findall(pattern, text_lower))
        
        # Deterioration
        for pattern in self.medical_patterns['deterioration']:
            indicators['severe'] += len(re.findall(pattern, text_lower))
        
        # Monitoring
        for pattern in self.medical_patterns['monitoring']:
            indicators['moderate'] += len(re.findall(pattern, text_lower))
        
        # Improvement
        for pattern in self.medical_patterns['improvement']:
            indicators['mild'] += len(re.findall(pattern, text_lower))
        
        return indicators
    
    def _aggregate_embeddings_by_phase(self,
                                      embeddings: List[np.ndarray],
                                      phases: List[str]) -> Dict[str, np.ndarray]:
        """Aggregate embeddings by temporal phase"""
        phase_embeddings = {
            'preoperative': [],
            'intraoperative': [],
            'postoperative': [],
            'all': []
        }
        
        for emb, phase in zip(embeddings, phases):
            if phase in phase_embeddings:
                phase_embeddings[phase].append(emb)
            phase_embeddings['all'].append(emb)
        
        # Calculate mean embeddings
        aggregated = {}
        for phase, emb_list in phase_embeddings.items():
            if emb_list:
                aggregated[phase] = np.mean(emb_list, axis=0)
            else:
                # Return zero vector if no embeddings
                embedding_dim = embeddings[0].shape[0] if embeddings else 768
                aggregated[phase] = np.zeros(embedding_dim)
        
        return aggregated
    
    def _generate_metadata(self, processed: Dict, phase: str) -> Dict:
        """Generate comprehensive metadata"""
        return {
            'requested_phase': phase,
            'num_notes': len(processed['cleaned_texts']),
            'total_words': sum(len(text.split()) for text in processed['cleaned_texts']),
            'total_sentences': sum(len(sents) for sents in processed['sentences']),
            'note_categories': dict(Counter(processed['categories'])),
            'temporal_phases': dict(Counter(processed['temporal_phases'])),
            'unique_entities': len(set([e['text'] for entities in processed['entities'] for e in entities])),
            'num_complication_mentions': sum(v['count'] for v in processed['complication_mentions'].values()),
            'severity_score': self._calculate_severity_score(processed['severity_indicators']),
            'has_section_embeddings': len(processed.get('section_embeddings', {})) > 0
        }
    
    def _calculate_severity_score(self, severity_indicators: Dict[str, int]) -> float:
        """Calculate overall severity score from indicators"""
        weights = {
            'critical': 3.0,
            'severe': 2.0,
            'moderate': 1.0,
            'mild': -0.5  # Improvement reduces severity
        }
        
        total_score = sum(count * weights[level] 
                         for level, count in severity_indicators.items())
        
        total_mentions = sum(severity_indicators.values())
        
        if total_mentions > 0:
            normalized_score = total_score / (total_mentions * 3)  # Normalize to [0, 1]
            return float(np.clip(normalized_score, 0, 1))
        
        return 0.0
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        embedding_dim = 768  # Default BERT dimension
        
        return {
            'raw_texts': [],
            'cleaned_texts': [],
            'embeddings': [],
            'entities': [],
            'categories': [],
            'temporal_phases': [],
            'temporal_positions': [],
            'sentences': [],
            'section_embeddings': {},
            'complication_mentions': {},
            'severity_indicators': {'critical': 0, 'severe': 0, 'moderate': 0, 'mild': 0},
            'aggregated_embeddings': {
                'preoperative': np.zeros(embedding_dim),
                'intraoperative': np.zeros(embedding_dim),
                'postoperative': np.zeros(embedding_dim),
                'all': np.zeros(embedding_dim)
            },
            'metadata': {
                'num_notes': 0,
                'total_words': 0,
                'severity_score': 0.0
            }
        }
    
    def extract_operative_details(self, notes_df: pd.DataFrame) -> Dict:
        """
        Extract specific operative details from intraoperative notes
        
        Returns:
            Dictionary with operative details
        """
        # Filter to intraoperative notes
        intraop_notes = notes_df[
            notes_df['CATEGORY'].str.lower().str.contains('|'.join(self.intraop_keywords), na=False)
        ]
        
        if intraop_notes.empty:
            return {}
        
        all_text = ' '.join(intraop_notes['TEXT'].astype(str)).lower()
        
        details = {}
        
        # Extract procedure type
        procedure_pattern = r'procedure[s]?:\s*([^\n]{10,150})'
        proc_matches = re.findall(procedure_pattern, all_text, re.IGNORECASE)
        if proc_matches:
            details['procedure'] = proc_matches[0].strip()
        
        # Extract operative time
        time_pattern = r'(?:operative time|procedure time|duration):\s*(\d+)\s*(?:hour|hr|min)'
        time_matches = re.findall(time_pattern, all_text)
        if time_matches:
            details['operative_time_minutes'] = int(time_matches[0])
        
        # Extract blood loss
        blood_pattern = r'(?:blood loss|ebl|estimated blood loss):\s*(\d+)\s*(?:ml|cc)'
        blood_matches = re.findall(blood_pattern, all_text)
        if blood_matches:
            details['blood_loss_ml'] = int(blood_matches[0])
        
        # Extract anesthesia type
        anesthesia_pattern = r'(?:anesthesia|anesthetic):\s*([^\n]{5,50})'
        anes_matches = re.findall(anesthesia_pattern, all_text)
        if anes_matches:
            details['anesthesia_type'] = anes_matches[0].strip()
        
        # Extract complications mentioned
        if 'complication' in all_text:
            comp_pattern = r'complications?:\s*([^\n]{10,200})'
            comp_matches = re.findall(comp_pattern, all_text)
            if comp_matches:
                details['intraoperative_complications'] = comp_matches[0].strip()
        
        return details