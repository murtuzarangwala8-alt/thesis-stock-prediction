import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def setup_plotting_theme():
    """Sets a unified, publication-ready seaborn theme."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True
    })

import re

def export_latex_table(df, filepath: Path, float_format="%.4f"):
    """Saves a DataFrame as a LaTeX table with escaped LaTeX characters."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            latex_str = df.to_latex(float_format=float_format, escape=True)
        except Exception:
            latex_str = df.to_latex(float_format=float_format)
            
        # Escape underscores and percent signs that are not already escaped
        latex_str = re.sub(r'(?<!\\)_', r'\\_', latex_str)
        latex_str = re.sub(r'(?<!\\)%', r'\\%', latex_str)
        
        with open(filepath, "w") as f:
            f.write(latex_str)
    except Exception as e:
        print(f"Warning: Failed to export LaTeX table to {filepath} (file may be locked): {e}")


def export_csv_table(df, filepath: Path):
    """Saves a DataFrame as a CSV table."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
    except Exception as e:
        print(f"Warning: Failed to export CSV table to {filepath} (file may be locked): {e}")
