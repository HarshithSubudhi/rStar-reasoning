from setuptools import setup, find_packages

setup(
    name="rstar",
    version="0.1.0",
    description="Python implementation of rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers (ICLR 2025)",
    author="HarshithSubudhi",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "llm": ["openai>=1.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
