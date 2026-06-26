from .mse import MSE
from .mae import MAE
from .arr import ARR
from .cr import CR
from .dd import DD
from .sor import SOR
from .sr import SR
from .vol import VOL
from .mdd import MDD
from .precision import Precision
from .recall import Recall
from .f1score import F1Score
from .accuracy import Accuracy
from .auc import AUC
from .hr import HitRatio
from .ic import IC
from .rankic import RANKIC
from .rankicir import RANKICIR

TRADING_METRICS = [
    ARR,
    CR,
    DD,
    SOR,
    SR,
    VOL,
    MDD,
]

CLASSIFICATION_METRICS = [
    Precision,
    Recall,
    F1Score,
    Accuracy,
    AUC
]

REGRESSION_METRICS = [
    MSE,
    MAE
]

RANK_METRICS = [
    IC,
    RANKIC,
    RANKICIR
]

OTHER_METRICS = [
    HitRatio
]

__all__ = [
    'MSE',
    'MAE',
    'IC',
    'RANKIC',
    'RANKICIR',
    'ARR',
    'CR',
    'DD',
    'SOR',
    'SR',
    'VOL',
    'MDD',
    'Precision',
    'Recall',
    'F1Score',
    'Accuracy',
    'AUC',
    'HitRatio'
]