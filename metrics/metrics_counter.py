import sys
from typing import Any

from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from containers import ClassificationMetrics, PredictionCounters


class MetricsCounter:
    def __init__(self) -> None:
        pass
    
    def __call__(self, c: PredictionCounters) -> ClassificationMetrics:
        return self.compute_metrics(c)

    @staticmethod
    def compute_metrics(c: PredictionCounters) -> ClassificationMetrics:
        m = ClassificationMetrics()
        m.precision = c.tp / (c.tp + c.fp) if c.tp != 0 else 0
        m.recall = c.tp / (c.tp + c.fn) if c.tp != 0  else 0
        m.specificity = c.tn / (c.tn + c.fp) if c.tn != 0 else 0
        
        m.f1 = 2.0 * m.precision * m.recall / (m.precision + m.recall) if m.precision + m.recall > 0 else 0
        if m.specificity * m.recall == 0:
            m.f3 = 0.0
        else:
            m.f3 = (10.0 * m.specificity * m.recall) / (m.specificity + m.recall * 9.0)
        
        m.apcer = m.frr = c.fp / (c.tn + c.fp) if c.fp != 0 else 0
        m.bpcer = m.far = c.fn / (c.tp + c.fn) if c.fn != 0 else 0
        m.acer = m.hter = (m.apcer + m.bpcer) / 2.0
 
        return m
