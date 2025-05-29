"""
Handles generation of a grid of roads from a grid of drawn in blocks
"""

from manim import *
from roads import *
import numpy as np
import sys
sys.path.append(".")
from Utils import misc_funcs

"""
patterns is a list of tuples
[0] of tuple is the 2d matrix for the pattern to recognise
[1] of tuple is a list of each block's data in the pattern

  the block's data is a dictionary:
    centre: a 2d tuple representing this block's location relative to the top left of the pattern
    lock: causes other patterns to be unable to overwrite the current pattern if placed, and no other patterns can match
      in the same place. this prevents situations where 2 valid patterns can overlap and not create a clear road path.
    type: Placeable class that is spawned in this position. If None, it is made empty.
    parameters: kwargs that the Placeable is initiated with
"""



patterns = [
    (
        [[1, 1]], [
            {"centre": (0, 0), "lock": False, "type": StraightRoad, "parameters": {"rotation": 90}},
            {"centre": (1, 0), "lock": False, "type": StraightRoad, "parameters": {"rotation": 90}},
            False
        ]
     ),
    (
        [[1],
         [1]], [
            {"centre": (0, 0), "lock": False, "type": StraightRoad, "parameters": {"rotation": 0}},
            {"centre": (0, 1), "lock": False, "type": StraightRoad, "parameters": {"rotation": 0}},
            True
        ]
     ),
    (
        [[1, 1],
         [1, 0]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 0}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 1}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 2}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 0, "section": 3}},
            False
        ]
     ),

    (
        [[1, 1],
         [0, 1]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 90, "section": 2}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 90, "section": 0}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 90, "section": 3}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 90, "section": 1}},
            True
        ]
     ),

    (
        [[0, 1],
         [1, 1]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 180, "section": 3}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 180, "section": 2}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 180, "section": 1}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 180, "section": 0}},
            True
        ]
     ),

    (
        [[1, 0],
         [1, 1]], [
            {"centre": (0, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 270, "section": 1}},
            {"centre": (1, 0), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 270, "section": 3}},
            {"centre": (0, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 270, "section": 0}},
            {"centre": (1, 1), "lock": False, "type": CurvedRoad, "parameters": {"rotation": 270, "section": 2}},
            True
        ]
     ),
    (
        [[1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]], [
            {"centre": (0, 0), "lock": False, "type": None},
            {"centre": (1, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 1}},
            {"centre": (2, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 2}},
            {"centre": (3, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 3}},
            {"centre": (0, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 4}},
            {"centre": (1, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 5}},
            {"centre": (2, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 6}},
            {"centre": (3, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 7}},
            {"centre": (0, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 8}},
            {"centre": (1, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 9}},
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (0, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 12}},
            {"centre": (1, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 0, "section": 13}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            False
        ]
    ),
    (
        [[1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1]], [
            {"centre": (2, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 12}},
            {"centre": (3, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 8}},
            {"centre": (4, 0), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 4}},
            {"centre": (5, 0), "lock": False, "type": None},
            {"centre": (2, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 13}},
            {"centre": (3, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 9}},
            {"centre": (4, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 5}},
            {"centre": (5, 1), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 1}},
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (4, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 6}},
            {"centre": (5, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 2}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            {"centre": (4, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 7}},
            {"centre": (5, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 90, "section": 3}},
            True
        ]
    ),
    (
        [[0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1]], [
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (4, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 13}},
            {"centre": (5, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 12}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            {"centre": (4, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 9}},
            {"centre": (5, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 8}},
            {"centre": (2, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 7}},
            {"centre": (3, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 6}},
            {"centre": (4, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 5}},
            {"centre": (5, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 4}},
            {"centre": (2, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 3}},
            {"centre": (3, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 2}},
            {"centre": (4, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 180, "section": 1}},
            {"centre": (5, 5), "lock": False, "type": None},
            True
        ]
    ),
    (
        [[1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1]], [
            {"centre": (0, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 3}},
            {"centre": (1, 2), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 7}},
            {"centre": (2, 2), "lock": False, "type": None},
            {"centre": (3, 2), "lock": False, "type": None},
            {"centre": (0, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 2}},
            {"centre": (1, 3), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 6}},
            {"centre": (2, 3), "lock": False, "type": None},
            {"centre": (3, 3), "lock": False, "type": None},
            {"centre": (0, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 1}},
            {"centre": (1, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 5}},
            {"centre": (2, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 9}},
            {"centre": (3, 4), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 13}},
            {"centre": (0, 5), "lock": False, "type": None},
            {"centre": (1, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 4}},
            {"centre": (2, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 8}},
            {"centre": (3, 5), "lock": True, "type": LargeCurvedRoad, "parameters": {"rotation": 270, "section": 12}},
            True
        ]
    ),
]

def get_pattern_data(pattern):
    """
    :param pattern: pattern matrix
    :return: data for this pattern, using the patterns list
    """
    for p in patterns:
        if np.array_equal(np.array(p[0]), np.array(pattern)):
            return p[1]
    return []


def generate_roads(map_grid, drawn_roads, animate, pattern_loc):
    """
    uses patterns to convert drawn roads to road objects
    :param drawn_roads: 2D matrix of 1s and 0s, where 1 is a drawn grid position
    :return: matrix of None or Road object
    """

    # iterate over the patterns to search for
    for pattern in map(lambda x: x[0], patterns):

        # get the data for blocks in this pattern
        pattern_data = get_pattern_data(pattern)

        if animate:
            map_grid._change_pattern_searcher(pattern, pattern_loc)

        # find grid locations (top left of pattern) where this pattern is found
        search_pattern(drawn_roads, pattern, pattern_data, map_grid, animate)


def search_pattern(drawn_roads, pattern, pattern_data, manim_mapgrid, animate):
    """
    :param drawn_roads: full matrix of 0s and 1s, where 1 is a drawn road
    :param pattern: matrix of pattern to search for
    :return: list of locations where the pattern occurs in the drawn roads. location is the top right of pattern.
    """
    pattern_size = np.array(pattern).shape[::-1]
    # iterate over every block in grid
    for row_num, row in enumerate(drawn_roads):
        if row_num > len(drawn_roads) - pattern_size[1]:
            break

        if manim_mapgrid is not None and animate:
            multiplier = 0.25 if pattern_data[-1] else 1
            if row_num:
                manim_mapgrid._move_pattern_searcher((row_num, 0), 0.5*multiplier)  # reset
            manim_mapgrid._move_pattern_searcher((row_num, len(row) - len(pattern[0])), 1.5*multiplier)  # end of line

        for col_num in range(len(row) - len(pattern[0])):

            # current block
            start = np.array([col_num, row_num])

            # get a matrix of the pattern's size, with top left location at 'start'
            compare_pattern = misc_funcs.cut_matrix(drawn_roads, start, start + pattern_size)

            # out of range if None
            if compare_pattern is None:
                continue

            # if the pattern matches
            if np.array_equal(np.array(compare_pattern), np.array(pattern)):
                location = (col_num, row_num)
                # for every block in the pattern
                for block in pattern_data[:-1]:
                    block: dict  # prevents warnings
                    # get the current block's grid location
                    block_location = (block["centre"][0] + location[0], block["centre"][1] + location[1])

                    if block["lock"]:
                        # if lock, prevent other blocks being placed here by setting that it was never drawn in,
                        # preventing patterns matching here
                        drawn_roads[block_location[1]][block_location[0]] = 0

                    manim_mapgrid.remove_road_block(block_location)

                    if block["type"] is not None:
                        # initiate road object with given parameters
                        parameters = block["parameters"]
                        parameters["grid_size"] = manim_mapgrid.grid_size
                        new_road = block["type"](**parameters)

                        manim_mapgrid.add_road_block(new_road, block_location)
