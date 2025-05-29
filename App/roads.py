from manim import *
import sys
sys.path.append(".")
from Utils.misc_funcs import rotate_vector_acw
from PIL import Image


class Road:
    image_path = ""

    def __init__(self, rotation, grid_size):
        self.rotation = rotation
        self.grid_size = grid_size
        self.mob = self._create_manim_image_mob()

    @property
    def section_image_path(self):
        return self.image_path

    def _create_manim_image_mob(self):
        image_obj = Image.open(self.section_image_path).convert("RGBA")
        array = np.array(image_obj)
        k = 4-int(self.rotation//90)
        rotated = np.rot90(array, k=k)
        im = ImageMobject(rotated, resampling_algorithm=0)  # nearest resampling

        return im

    def collision(self, rel_point):
        raise NotImplementedError


class StraightRoad(Road):
    image_path = "assets/straight_road.png"

    def collision(self, rel_point):
        return True  # anywhere on space is road


def circle(x, radius):
    """
    Equation for positive y values of a circle, starting at 0,0
    sqrt(r^2 - (x-r)^2)
    :param x: x in circle equaion
    :param radius: circle radius

    :returns: positive y value at x, or NAN if invalid
    """
    a = np.square(radius) - np.square(x-radius)
    if a < 0:
        return np.nan
    return np.sqrt(a)


def circle_collision_road(relative_point, grid_location, grid_size, rotation, width):
    """
    :param relative_point: relative point to individual grid square location
    :param grid_location: grid location, to make relative_point relative to entire road section
    :param grid_size: size of grid square
    :param rotation: rotation of curved road
    :param width: width of entire curved road (in grid squares)
    :return: true or false if point is outside of road
    """

    """
    rotate relative point about center of grid square. then we can use relative point ignoring rotation.
    relative point can then be processed as if the road is like this
    ░░░░░░░░█████
    ░░░░░███░░░░░
    ░░░██░░░░░░░░
    ░██░░░░░░████
    ██░░░░░██░░░░
    █░░░░██░░░░░░
    """
    relative_point = rotate_vector_acw(relative_point - grid_size/2, -rotation) + grid_size/2

    # point relative to top left of curved road
    new_point = np.array(relative_point) + np.array(grid_location) * grid_size

    outer_circle = new_point[1] > (width*grid_size-circle(new_point[0], width*grid_size))  # true if on road (below outer circle)

    inner_circle_height = width*grid_size - circle(new_point[0] - grid_size, (width-1)*grid_size)

    inner_circle = True  # default to true ( < inner circle), then if valid, and greater than, change to false
    if not np.isnan(inner_circle_height):
        inner_circle = new_point[1] < inner_circle_height

    return inner_circle and outer_circle


class CurvedRoad(Road):
    image_paths = [
        "assets/curve_road_1.png",
        "assets/curve_road_2.png",
        "assets/curve_road_3.png",
        "assets/curve_road_4.png",
    ]
    width = 2

    def __init__(self, rotation, grid_size, section):
        self.section = section
        super().__init__(rotation, grid_size)

    @property
    def grid_location(self):
        """
        :return: Grid location tuple, relative to top left of curved road pattern
        """
        return self.section % self.width, self.section // self.width

    @property
    def section_image_path(self):
        return self.image_paths[self.section]

    def collision(self, rel_point, size_override=None):
        if size_override is None:
            size = self.grid_size
        else:
            size = size_override
        return circle_collision_road(rel_point, self.grid_location, size, self.rotation, self.width)


class LargeCurvedRoad(CurvedRoad):
    # None means the block is empty
    image_paths = [
        None,
        "assets/large_curve_road/section_1.png",
        "assets/large_curve_road/section_2.png",
        "assets/large_curve_road/section_3.png",
        "assets/large_curve_road/section_4.png",
        "assets/large_curve_road/section_5.png",
        "assets/large_curve_road/section_6.png",
        "assets/large_curve_road/section_7.png",
        "assets/large_curve_road/section_8.png",
        "assets/large_curve_road/section_9.png",
        None,
        None,
        "assets/large_curve_road/section_12.png",
        "assets/large_curve_road/section_13.png",
        None,
        None
    ]

    width = 4
