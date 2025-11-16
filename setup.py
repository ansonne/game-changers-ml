from setuptools import setup, find_packages

setup(
    name="game_changers_ml",
    version="1.0.0",
    description="Machine Learning system for Game Changers 2025 predictions",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "requests==2.31.0",
        "pytest==7.4.0",
        "python-dotenv==1.0.0",
        "pyyaml==6.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
    ],
    python_requires=">=3.8",
    package_dir={'': '.'},
)