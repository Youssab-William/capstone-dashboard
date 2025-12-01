"""
Politeness Analysis Module

Uses validated politeness features from Stanford/Cornell research.
Based on Brown & Levinson Politeness Theory with computational linguistics validation.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, Any, List, Tuple


class PolitenessAnalyzer:
    """Validated politeness features analyzer."""

    def __init__(self):
        """Initialize the politeness analyzer with validated patterns."""
        self.positive_politeness_markers = {
            'Gratitude': [
                r'\b(?:thank you|thanks|grateful|appreciate|much appreciated|thank you so much)\b',
            ],
            'Greeting': [
                r'\b(?:hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
            ],
            'Positive_Lexicon': [
                r'\b(?:great|excellent|wonderful|fantastic|amazing|perfect|brilliant|outstanding)\b',
            ]
        }

        self.negative_politeness_markers = {
            'Please_Markers': [
                r'\b(?:please|kindly|could you please|would you please|if you could)\b',
            ],
            'Please_Start': [
                r'^please\b',
            ],
            'Modal_Hedges': [
                r'\b(?:could you|would you|might you|may i|can i|would it be possible)\b',
            ],
            'Hedges': [
                r'\b(?:maybe|perhaps|possibly|might|could|may|i think|i believe|likely|probably|sort of|kind of)\b',
            ],
            'Apologizing': [
                r'\b(?:sorry|apologies|apologize|excuse me|pardon me|my apologies|i\'m sorry)\b',
            ],
            'Deference': [
                r'\b(?:if you don\'t mind|when convenient|at your convenience|if possible|when you have time)\b',
            ],
            'Indirect_Questions': [
                r'\b(?:i was wondering if|do you think|would you say|is it possible that|could it be)\b',
            ]
        }

        self.impoliteness_markers = {
            'Direct_Commands': [
                r'^(?:do|make|get|give|tell|show|explain|describe|list|name|provide)\b',
            ],
            'Second_Person_Start': [
                r'^you\b',
            ]
        }

        # Compile all patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.compiled_patterns = {}

        for category in [self.positive_politeness_markers, self.negative_politeness_markers, self.impoliteness_markers]:
            for strategy, patterns in category.items():
                self.compiled_patterns[strategy] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze politeness features in a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with politeness analysis results
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {
                'score': 3.0,  # Neutral score
                'feature_count': 0,
                'strategies': '',
                'confidence': 0.0
            }

        text = text.strip()
        detected_strategies = {}
        total_features = 0

        # Check each politeness strategy
        for strategy, patterns in self.compiled_patterns.items():
            count = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                count += len(matches)

            if count > 0:
                detected_strategies[strategy] = count
                total_features += count

        # Calculate politeness score (1-5 scale)
        score = self._calculate_politeness_score(detected_strategies, len(text))

        # Format strategies string
        strategies_str = '; '.join([f"{strategy}:{count}" for strategy, count in detected_strategies.items()])

        # Calculate confidence based on number and strength of features
        confidence = min(1.0, total_features * 0.2)

        return {
            'score': score,
            'feature_count': total_features,
            'strategies': strategies_str,
            'confidence': confidence
        }

    def _calculate_politeness_score(self, strategies: Dict[str, int], text_length: int) -> float:
        """
        Calculate overall politeness score based on detected strategies.

        Args:
            strategies: Dictionary of detected strategies and counts
            text_length: Length of the text

        Returns:
            Politeness score on 1-5 scale
        """
        base_score = 3.0  # Neutral

        # Positive contributions
        positive_weight = 0
        if 'Gratitude' in strategies:
            positive_weight += strategies['Gratitude'] * 0.5
        if 'Greeting' in strategies:
            positive_weight += strategies['Greeting'] * 0.3
        if 'Positive_Lexicon' in strategies:
            positive_weight += strategies['Positive_Lexicon'] * 0.2

        # Negative politeness contributions (also positive for overall politeness)
        negative_politeness_weight = 0
        for strategy in ['Please_Markers', 'Please_Start', 'Modal_Hedges', 'Hedges', 'Apologizing', 'Deference', 'Indirect_Questions']:
            if strategy in strategies:
                weight = 0.3 if strategy in ['Please_Markers', 'Please_Start'] else 0.2
                negative_politeness_weight += strategies[strategy] * weight

        # Impoliteness penalties
        impoliteness_penalty = 0
        if 'Direct_Commands' in strategies:
            impoliteness_penalty += strategies['Direct_Commands'] * 0.3
        if 'Second_Person_Start' in strategies:
            impoliteness_penalty += strategies['Second_Person_Start'] * 0.1

        # Calculate final score
        total_positive = positive_weight + negative_politeness_weight

        # Normalize by text length (longer texts might have more markers)
        if text_length > 0:
            normalized_positive = total_positive * (100 / max(text_length, 100))
            normalized_penalty = impoliteness_penalty * (100 / max(text_length, 100))
        else:
            normalized_positive = 0
            normalized_penalty = 0

        final_score = base_score + normalized_positive - normalized_penalty

        # Clamp to 1-5 range
        return max(1.0, min(5.0, final_score))

    def analyze_dataframe(self, df: pd.DataFrame,
                         prompt_col: str = 'PromptText',
                         response_col: str = 'ResponseText') -> pd.DataFrame:
        """
        Analyze politeness for all prompts and responses in a dataframe.

        Args:
            df: Input dataframe
            prompt_col: Column name containing prompts
            response_col: Column name containing responses

        Returns:
            Dataframe with added politeness columns
        """
        result_df = df.copy()

        # Analyze prompts
        if prompt_col in df.columns:
            print(f"Analyzing politeness for {len(df)} prompts...")
            prompt_analyses = df[prompt_col].apply(self.analyze_text)

            result_df['Prompt_ValidatedPolitenessScore'] = [analysis['score'] for analysis in prompt_analyses]
            result_df['Prompt_ValidatedFeatureCount'] = [analysis['feature_count'] for analysis in prompt_analyses]
            result_df['Prompt_ValidatedStrategies'] = [analysis['strategies'] for analysis in prompt_analyses]
            result_df['Prompt_ValidatedConfidence'] = [analysis['confidence'] for analysis in prompt_analyses]

        # Analyze responses
        if response_col in df.columns:
            print(f"Analyzing politeness for {len(df)} responses...")
            response_analyses = df[response_col].apply(self.analyze_text)

            result_df['Response_ValidatedPolitenessScore'] = [analysis['score'] for analysis in response_analyses]
            result_df['Response_ValidatedFeatureCount'] = [analysis['feature_count'] for analysis in response_analyses]
            result_df['Response_ValidatedStrategies'] = [analysis['strategies'] for analysis in response_analyses]
            result_df['Response_ValidatedConfidence'] = [analysis['confidence'] for analysis in response_analyses]

        return result_df

    def get_politeness_label(self, politeness_score: float) -> str:
        """
        Convert politeness score to categorical label.

        Args:
            politeness_score: Politeness score (1-5 scale)

        Returns:
            Politeness label
        """
        if politeness_score >= 4.0:
            return 'very_polite'
        elif politeness_score >= 3.5:
            return 'polite'
        elif politeness_score >= 2.5:
            return 'neutral'
        elif politeness_score >= 2.0:
            return 'impolite'
        else:
            return 'very_impolite'