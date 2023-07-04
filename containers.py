from dataclasses import dataclass


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