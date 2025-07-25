"""Setup configuration for UltSignalTrader."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ultsignaltrader",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade crypto trading bot framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ultsignaltrader",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ccxt>=4.0.0",
        "ta>=0.10.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "nbformat>=5.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultsignaltrader=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ultsignaltrader": ["*.yaml", "*.yml"],
    },
)