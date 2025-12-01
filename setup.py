from setuptools import setup, find_packages

setup(
    name="capstone-pipeline",
    version="1.0.0",
    description="LLM tone analysis pipeline",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "google-generativeai>=0.3.0",
        "requests>=2.31.0",
        "vaderSentiment>=3.3.2",
        "detoxify>=0.5.0",
    ],
)

