from dataclasses import dataclass

import torch


@dataclass
class Rect2i:
    """
    Rectangle. Analagous to OpenCV Rect2i
    """
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x1(self) -> int:
        return int(self.x)
    
    @property
    def x2(self) -> int:
        return int(self.x + self.width)
    
    @property
    def y1(self) -> int:
        return int(self.y)
    
    @property
    def y2(self) -> int:
        return int(self.y + self.height)
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def top(self) -> int:
        return self.y
    
    @property
    def right(self) -> int:
        return int(self.x + self.width)
    
    @property
    def bottom(self) -> int:
        return int(self.y + self.height)
    
    @property
    def area(self) -> int:
        return int(self.width * self.height)
    

@dataclass
class ClassificationMetrics:
    precision: float = 0.0
    recall: float = 0.0  # == sensitivity
    specificity: float = 0.0
    f1: float = 0.0
    f3: float = 0.0
    apcer: float = 0.0
    bpcer: float = 0.0
    acer: float = 0.0
    far: float = 0.0
    frr: float = 0.0
    hter: float = 0.0
    threshold: float = 0.0
    
    def __repr__(self) -> str:
        text = (f"APCER: {self.apcer*100:.2f}%, "
                f"BPCER: {self.bpcer*100:.2f}%, "
                f"ACER: {self.acer*100:.2f}%, "
                f"F1: {self.f1*100:.2f}%, "
                f"F3: {self.f3*100:.2f}%, "
                f"P: {self.precision*100:.2f}%, "
                f"R: {self.recall*100:.2f}%, "
                f"S: {self.specificity*100:.2f}%"
                )
        return text


@dataclass
class PredictionCounters:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    
    def __add__(self, other):
        return PredictionCounters(
            tp = self.tp + other.tp,
            fp = self.fp + other.fp,
            tn = self.tn + other.tn,
            fn = self.fn + other.fn
        )
    
    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor((self.tp, self.fp, self.tn, self.fn))
    
    @staticmethod    
    def from_tensor(tensor: torch.Tensor):
        assert len(tensor) == 4
        return PredictionCounters(
            tp = tensor[0],
            fp = tensor[1],
            tn = tensor[2],
            fn = tensor[3]
        )