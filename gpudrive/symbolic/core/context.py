from enum import Enum

class RoadContext(Enum):
    """Enum for different road contexts an agent can be in."""
    IN_INTERSECTION = "IN_INTERSECTION"
    IN_LANE = "IN_LANE"
    NEAR_CROSSWALK = "NEAR_CROSSWALK"
    NEAR_STOP_SIGN = "NEAR_STOP_SIGN"

    def __str__(self):
        return self.value 