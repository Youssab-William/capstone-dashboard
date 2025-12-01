"""
Sentiment Analysis Module

Uses VADER sentiment analysis for robust sentiment detection in conversational text.
Based on validated methodology from computational social science research.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """VADER-based sentiment analyzer for prompt and response text."""

    def __init__(self):
        """Initialize the VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0
            }

        try:
            scores = self.analyzer.polarity_scores(text)
            return scores
        except Exception:
            # Return neutral scores on error
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0
            }

    def analyze_dataframe(self, df: pd.DataFrame,
                         prompt_col: str = 'PromptText',
                         response_col: str = 'ResponseText') -> pd.DataFrame:
        """
        Analyze sentiment for all prompts and responses in a dataframe.

        Args:
            df: Input dataframe
            prompt_col: Column name containing prompts
            response_col: Column name containing responses

        Returns:
            Dataframe with added sentiment columns
        """
        result_df = df.copy()

        # Analyze prompts
        if prompt_col in df.columns:
            prompt_scores = df[prompt_col].apply(self.analyze_text)
            result_df['Prompt_SentimentScore'] = [score['compound'] for score in prompt_scores]
            result_df['Prompt_SentimentPos'] = [score['pos'] for score in prompt_scores]
            result_df['Prompt_SentimentNeu'] = [score['neu'] for score in prompt_scores]
            result_df['Prompt_SentimentNeg'] = [score['neg'] for score in prompt_scores]

        # Analyze responses
        if response_col in df.columns:
            response_scores = df[response_col].apply(self.analyze_text)
            result_df['Response_SentimentScore'] = [score['compound'] for score in response_scores]
            result_df['Response_SentimentPos'] = [score['pos'] for score in response_scores]
            result_df['Response_SentimentNeu'] = [score['neu'] for score in response_scores]
            result_df['Response_SentimentNeg'] = [score['neg'] for score in response_scores]

        return result_df

    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Convert compound sentiment score to categorical label.

        Args:
            compound_score: VADER compound score (-1 to 1)

        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'