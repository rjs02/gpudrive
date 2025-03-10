from enum import Enum

class SpatialRelation(Enum):
    """Enum for different spatial relations between agents."""
    AHEAD = "AHEAD"
    BEHIND = "BEHIND"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    APPROACHING = "APPROACHING"
    YIELDING = "YIELDING"

    def __str__(self):
        return self.value 