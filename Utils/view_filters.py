"""
Classes used to filter which elements should be displayed on screen

Types:

ai_rays
ai_ray_collisions

grid

collision

q_target_change
"""

FILTERS = []


def can_show_type(draw_type):
    bools = []

    for f in FILTERS:
        bools.append(f.show_type(draw_type))

    return all(bools)


class Filter:
    def show_type(self, type):
        pass


class AllowedFilter(Filter):
    allowed = []

    def show_type(self, draw_type):
        # Check for exact matches
        if draw_type in self.allowed:
            return True
        
        # Check for pattern matches (substring at beginning) 
        for allowed_type in self.allowed:
            if allowed_type.endswith('*') and draw_type.startswith(allowed_type[:-1]):
                return True
                
        return False


class BlockedFilter(Filter):
    blocked = []

    def show_type(self, draw_type):
        # Check for exact matches
        if draw_type in self.blocked:
            return False
            
        # Check for pattern matches (substring at beginning)
        for blocked_type in self.blocked:
            if blocked_type.endswith('*') and draw_type.startswith(blocked_type[:-1]):
                return False
                
        return True


class NoCollision(BlockedFilter):
    blocked = ["collision"]


class NoGrid(BlockedFilter):
    blocked = ["grid"]


class NoAiVis(BlockedFilter):
    blocked = ["ai_rays", "ai_ray_collisions"]


class AiVisOnly(AllowedFilter):
    def __init__(self, rays=False):
        self.allowed = ["ai_ray_collisions", "car"]
        if rays:
            self.allowed.append("ai_rays")


class FastTrainingView(AllowedFilter):
    """
    Allows training visualization with full track display
    """
    def __init__(self):
        # Include all necessary elements for clear visualization
        self.allowed = [
            "ai_rays", 
            "car",
            "grid",           # Added grid to show grid lines
            "straight_road",
            "curved_road_*"   # Match all curved road types
        ]
