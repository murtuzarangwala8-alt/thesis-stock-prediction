"""
TFDMGA — Temporal Fusion Deep Multimodal Gated Attention Network
Package initialisation.
"""
from .config import TFDMGAConfig, WALK_FORWARD_FOLDS, TEST_YEARS
from .model import TFDMGA, build_model
from .dataset import MasterDataStore, FinancialPanelDataset, WalkForwardSplitter, make_dataloader
from .losses import MultiTaskLoss, HuberLoss, RankingLoss, ICLoss
from .metrics import evaluate_predictions, format_metrics_table
from .trainer import Trainer
from .walkforward import WalkForwardEngine
from .evaluate import Evaluator
from .utils import set_seed, setup_logger, count_parameters

__version__ = "1.0.0"
__author__  = "TFDMGA Research Framework"

__all__ = [
    "TFDMGAConfig",
    "WALK_FORWARD_FOLDS",
    "TEST_YEARS",
    "TFDMGA",
    "build_model",
    "MasterDataStore",
    "FinancialPanelDataset",
    "WalkForwardSplitter",
    "make_dataloader",
    "MultiTaskLoss",
    "HuberLoss",
    "RankingLoss",
    "ICLoss",
    "evaluate_predictions",
    "format_metrics_table",
    "Trainer",
    "WalkForwardEngine",
    "Evaluator",
    "set_seed",
    "setup_logger",
    "count_parameters",
]
