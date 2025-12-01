#!/usr/bin/env python3
"""
Main Entry Point for Phase 2: Comprehensive Metrics Analysis Pipeline

This script provides a single entry point to analyze CSV files containing prompt/response data
and generate comprehensive metrics including sentiment, toxicity, politeness, and refusal detection.

Usage:
    python run_metrics_pipeline.py input.csv output.csv [--config config.json]

Input CSV Format (minimum required columns):
    - PromptText: The original prompt/question
    - ResponseText: The model's response
    - Model: The model name (optional, for analysis)

Output: Enhanced CSV with comprehensive metrics as documented in the metrics documentation.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our metrics modules
from phase2_metrics_analysis.modules.sentiment_analyzer import SentimentAnalyzer
from phase2_metrics_analysis.modules.toxicity_analyzer import ToxicityAnalyzer
from phase2_metrics_analysis.modules.politeness_analyzer import PolitenessAnalyzer
from phase2_metrics_analysis.modules.refusal_disclaimer_detector import RefusalDisclaimerDetector
from utils.data_validation import validate_input_csv, validate_output_csv


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('metrics_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path=None):
    """Load configuration file or use defaults."""
    default_config = {
        "sentiment_analyzer": {
            "enabled": True,
            "tool": "vader"
        },
        "toxicity_analyzer": {
            "enabled": True,
            "tool": "roberta",
            "model_name": "unitary/unbiased-toxic-roberta"
        },
        "politeness_analyzer": {
            "enabled": True,
            "tool": "validated_features"
        },
        "refusal_disclaimer_detector": {
            "enabled": True,
            "tool": "rule_based"
        },
        "processing": {
            "batch_size": 100,
            "deduplicate_prompts": True,
            "save_intermediate": True
        }
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            # Merge with defaults
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value

    return default_config


def initialize_analyzers(config):
    """Initialize all metric analyzers based on configuration."""
    analyzers = {}

    if config["sentiment_analyzer"]["enabled"]:
        analyzers["sentiment"] = SentimentAnalyzer()

    if config["toxicity_analyzer"]["enabled"]:
        analyzers["toxicity"] = ToxicityAnalyzer(
            model_name=config["toxicity_analyzer"]["model_name"]
        )

    if config["politeness_analyzer"]["enabled"]:
        analyzers["politeness"] = PolitenessAnalyzer()

    if config["refusal_disclaimer_detector"]["enabled"]:
        analyzers["refusal_disclaimer"] = RefusalDisclaimerDetector()

    return analyzers


def process_data(df, analyzers, config, logger):
    """Process the dataframe with all enabled analyzers."""
    logger.info(f"Processing {len(df)} rows with {len(analyzers)} analyzers")

    # Validate required columns
    required_columns = ['PromptText', 'ResponseText']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Initialize result dataframe
    result_df = df.copy()

    # Process each analyzer
    for analyzer_name, analyzer in analyzers.items():
        logger.info(f"Running {analyzer_name} analyzer...")

        try:
            if analyzer_name == "sentiment":
                # Analyze both prompts and responses
                result_df = analyzer.analyze_dataframe(result_df,
                                                     prompt_col='PromptText',
                                                     response_col='ResponseText')

            elif analyzer_name == "toxicity":
                # Multi-dimensional toxicity analysis
                result_df = analyzer.analyze_dataframe(result_df,
                                                     prompt_col='PromptText',
                                                     response_col='ResponseText')

            elif analyzer_name == "politeness":
                # Validated politeness features
                result_df = analyzer.analyze_dataframe(result_df,
                                                     prompt_col='PromptText',
                                                     response_col='ResponseText')

            elif analyzer_name == "refusal_disclaimer":
                # Rule-based refusal and disclaimer detection
                result_df = analyzer.analyze_dataframe(result_df,
                                                     prompt_col='PromptText',
                                                     response_col='ResponseText')

            logger.info(f"✓ {analyzer_name} analysis completed")

        except Exception as e:
            logger.error(f"Error in {analyzer_name} analyzer: {str(e)}")
            # Continue with other analyzers
            continue

    return result_df


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='LLM Metrics Analysis Pipeline')
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_csv', help='Output CSV file path')
    parser.add_argument('--config', help='Configuration JSON file path')
    parser.add_argument('--validate', action='store_true', help='Run validation checks')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("Starting LLM Metrics Analysis Pipeline")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded: {len([k for k, v in config.items() if isinstance(v, dict) and v.get('enabled', False)])} analyzers enabled")

        # Load input data
        logger.info(f"Loading input data from {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Validate input if requested
        if args.validate:
            validate_input_csv(df, logger)

        # Initialize analyzers
        analyzers = initialize_analyzers(config)
        logger.info(f"Initialized {len(analyzers)} analyzers")

        # Process data
        result_df = process_data(df, analyzers, config, logger)

        # Save results
        logger.info(f"Saving results to {args.output_csv}")
        result_df.to_csv(args.output_csv, index=False)

        # Validate output if requested
        if args.validate:
            validate_output_csv(result_df, logger)

        logger.info(f"✓ Pipeline completed successfully")
        logger.info(f"Output: {len(result_df)} rows with {len(result_df.columns)} columns")

        # Print summary
        print("\n" + "="*60)
        print("METRICS ANALYSIS PIPELINE COMPLETED")
        print("="*60)
        print(f"Input file: {args.input_csv}")
        print(f"Output file: {args.output_csv}")
        print(f"Rows processed: {len(result_df)}")
        print(f"Columns added: {len(result_df.columns) - len(df.columns)}")
        print(f"Total columns: {len(result_df.columns)}")
        print("="*60)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()