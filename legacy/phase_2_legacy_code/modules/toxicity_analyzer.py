"""
Toxicity Analysis Module

Uses RoBERTa-based unbiased toxic model for multi-dimensional toxicity detection.
Based on Unitary AI's validated model with bias mitigation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    print("Warning: detoxify not available. Install with: pip install detoxify")


class ToxicityAnalyzer:
    """RoBERTa-based toxicity analyzer for multi-dimensional toxicity detection."""

    def __init__(self, model_name: str = "unbiased"):
        """
        Initialize the toxicity analyzer.

        Args:
            model_name: Model to use ('unbiased', 'original', 'multilingual')
        """
        self.model_name = model_name
        self.model = None

        if DETOXIFY_AVAILABLE:
            try:
                self.model = Detoxify(model_name)
            except Exception as e:
                print(f"Error loading detoxify model: {e}")
                self.model = None

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze toxicity of a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with toxicity scores
        """
        if not DETOXIFY_AVAILABLE or self.model is None:
            # Return default values if model not available
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
                'identity_attack': 0.0
            }

        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
                'identity_attack': 0.0
            }

        try:
            scores = self.model.predict(text)
            # Convert numpy arrays to floats
            return {key: float(value) for key, value in scores.items()}
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
                'identity_attack': 0.0
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze toxicity for a batch of texts (more efficient).

        Args:
            texts: List of texts to analyze

        Returns:
            List of dictionaries with toxicity scores
        """
        if not DETOXIFY_AVAILABLE or self.model is None:
            return [self.analyze_text("") for _ in texts]

        # Filter out invalid texts
        valid_texts = []
        text_indices = []

        for i, text in enumerate(texts):
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                continue
            valid_texts.append(text)
            text_indices.append(i)

        if not valid_texts:
            return [self.analyze_text("") for _ in texts]

        try:
            # Analyze valid texts in batch
            batch_scores = self.model.predict(valid_texts)

            # Create results list
            results = [self.analyze_text("") for _ in texts]

            # Fill in results for valid texts
            for batch_idx, text_idx in enumerate(text_indices):
                scores = {key: float(values[batch_idx]) for key, values in batch_scores.items()}
                results[text_idx] = scores

            return results

        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return [self.analyze_text(text) for text in texts]

    def analyze_dataframe(self, df: pd.DataFrame,
                         prompt_col: str = 'PromptText',
                         response_col: str = 'ResponseText',
                         batch_size: int = 100) -> pd.DataFrame:
        """
        Analyze toxicity for all prompts and responses in a dataframe.

        Args:
            df: Input dataframe
            prompt_col: Column name containing prompts
            response_col: Column name containing responses
            batch_size: Batch size for processing

        Returns:
            Dataframe with added toxicity columns
        """
        result_df = df.copy()

        # Analyze prompts
        if prompt_col in df.columns:
            print(f"Analyzing toxicity for {len(df)} prompts...")
            prompt_texts = df[prompt_col].tolist()

            # Process in batches
            prompt_scores = []
            for i in range(0, len(prompt_texts), batch_size):
                batch_texts = prompt_texts[i:i+batch_size]
                batch_scores = self.analyze_batch(batch_texts)
                prompt_scores.extend(batch_scores)

            # Add columns
            for metric in ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']:
                col_name = f'RoBERTa_Prompt_{metric.title().replace("_", "")}Score'
                result_df[col_name] = [score[metric] for score in prompt_scores]

        # Analyze responses
        if response_col in df.columns:
            print(f"Analyzing toxicity for {len(df)} responses...")
            response_texts = df[response_col].tolist()

            # Process in batches
            response_scores = []
            for i in range(0, len(response_texts), batch_size):
                batch_texts = response_texts[i:i+batch_size]
                batch_scores = self.analyze_batch(batch_texts)
                response_scores.extend(batch_scores)

            # Add columns
            for metric in ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']:
                col_name = f'RoBERTa_Response_{metric.title().replace("_", "")}Score'
                result_df[col_name] = [score[metric] for score in response_scores]

        return result_df

    def get_toxicity_label(self, toxicity_score: float, threshold: float = 0.5) -> str:
        """
        Convert toxicity score to categorical label.

        Args:
            toxicity_score: Toxicity score (0 to 1)
            threshold: Threshold for toxic classification

        Returns:
            Label: 'toxic' or 'non_toxic'
        """
        return 'toxic' if toxicity_score >= threshold else 'non_toxic'