"""
Miscellaneous functions that are used in multiple app screens
"""

import sys
sys.path.append(".")
from Utils import global_settings as gs
import pygame
from Utils import view_filters


def draw_background(screen, grid=True):
    # Background is drawn by world's placeables, but this is beneath
    screen.fill(gs.COL_BACKGROUND)
    
    # Add a subtle gradient effect to the background
    gradient_surface = pygame.Surface((gs.WIDTH, gs.HEIGHT), pygame.SRCALPHA)
    for y in range(0, gs.HEIGHT, 4):
        # Create darkening gradient from top to bottom
        darkness = max(0, min(40, int(y * 40 / gs.HEIGHT)))
        gradient_color = (0, 0, 0, darkness)
        pygame.draw.line(gradient_surface, gradient_color, (0, y), (gs.WIDTH, y), 4)
    
    screen.blit(gradient_surface, (0, 0))

    # Only draw grid if the filter allows it
    if not view_filters.can_show_type("grid") or not grid:
        return

    # Draw Grid with improved visualization
    grid_color = gs.COL_GRID
    
    # Draw thicker lines for major grid sections
    major_interval = 5
    
    for x in range((gs.WIDTH // round(gs.GRID_SIZE_PIXELS)) + 1):
        line_thickness = 2 if x % major_interval == 0 else 1
        line_color = (grid_color[0], grid_color[1], grid_color[2], 
                     255 if x % major_interval == 0 else 180)
        pygame.draw.line(screen, line_color, 
                         (x * gs.GRID_SIZE_PIXELS, 0), 
                         (x * gs.GRID_SIZE_PIXELS, gs.HEIGHT), 
                         line_thickness)
    
    for y in range((gs.HEIGHT // round(gs.GRID_SIZE_PIXELS)) + 1):
        line_thickness = 2 if y % major_interval == 0 else 1
        line_color = (grid_color[0], grid_color[1], grid_color[2], 
                     255 if y % major_interval == 0 else 180)
        pygame.draw.line(screen, line_color, 
                         (0, y * gs.GRID_SIZE_PIXELS), 
                         (gs.WIDTH, y * gs.GRID_SIZE_PIXELS), 
                         line_thickness)