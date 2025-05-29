from manim import *
from typing import Iterable, Union, Optional, Tuple
import sys
sys.path.append(".")
from Utils.misc_funcs import cut_matrix
from App.road_generation import generate_roads as generate_roads_from_patterns  # make different to method name
from App.roads import Road


def get_mob_or_none(road):
    try:
        return road.mob
    except AttributeError:
        return None


class MapGrid:
    def __init__(self, scene: Scene, dimensions=(16, 9), grid_size=0.7, grid_kwargs=None, initial_pos=(0, 0)):
        if grid_kwargs is None:
            grid_kwargs = {}
        self.contained_mobjects = []

        self.scene = scene
        self.grid_size = grid_size

        self.matrix = np.full(shape=dimensions, fill_value=None).tolist()  # 2d array of None and Road objects
        self.table = self._create_initial_grid_mobject(dimensions, grid_kwargs)
        self.table.shift(initial_pos + (0,))
        scene.play(Write(self.table), run_time=1.5)

        self.draw_solid_background()

        self.pattern_searcher = None

    def _create_initial_grid_mobject(self, dimensions, grid_kwargs) -> MobjectTable:
        data = [[Square(color=BLACK, side_length=self.grid_size) for _ in range(dimensions[0])] for _ in range(dimensions[1])]
        return MobjectTable(data, include_outer_lines=True, h_buff=0, v_buff=0, **grid_kwargs)

    def draw_solid_background(self, color=GREEN, animate=True):
        cells = []
        for row in range(self.table.row_dim):
            for col in range(self.table.col_dim):
                if not animate:
                    self.table.add_highlighted_cell((row + 1, col + 1), color=color)
                else:
                    cell = self.table.get_highlighted_cell((row + 1, col + 1), color=color)
                    cell.scale(0.001)
                    cells.append(cell)
                    self.contained_mobjects.append(cell)

        if cells:
            self.scene.play(
                AnimationGroup(
                    *(ScaleInPlace(cell, 950) for cell in cells),
                    lag_ratio=0.01,
                    run_time=2
                )
            )

    def draw_roads(self, road_matrix: Iterable[Iterable[int]], animation_time=None):
        bg_cells = []
        for row_num, row in enumerate(road_matrix):
            for col_num, col in enumerate(row):
                if col == 1:
                    cell = self.table.get_highlighted_cell((row_num + 1, col_num + 1), GREY)
                    self.contained_mobjects.append(cell)
                    bg_cells.append(cell)

        if animation_time is not None:
            self.scene.play(
                AnimationGroup(
                    *(Write(cell) for cell in bg_cells),
                    lag_ratio=0.5,
                    run_time=animation_time
                )
            )
        else:
            # dont animate
            self.scene.add(*bg_cells)
        return bg_cells

    def remove_cells(self, cells):
        self.scene.play(
            AnimationGroup(
                *(obj.animate.set_opacity(0) for obj in cells),
                lag_ratio=0.1,
                run_time=1
            ),
        )
        for cell in cells:
            self.contained_mobjects.remove(cell)

    def _change_pattern_searcher(self, pattern, pattern_loc):
        """
        Called when animating pattern searching
        Changes the displayed pattern of the 'searcher'
        :param pattern_loc: position on screen of the key
        """
        if self.pattern_searcher is not None:
            self.scene.play(Unwrite(self.pattern_searcher), Unwrite(self.pattern_searcher.key), run_time=0.2)

        pattern = np.array(pattern)

        cells = []
        for row_n, row in enumerate(pattern):
            for col_n, col in enumerate(row):
                cell = self.table.get_highlighted_cell((row_n+1, col_n+1), color=ORANGE)
                cell.set_opacity(0.8 if col else 0)
                cells.append(cell)

        self.pattern_searcher = VGroup(*cells)
        self.pattern_searcher.arrange_in_grid(len(pattern[:,0]), len(pattern[0]), buff=0)

        key = self.pattern_searcher.copy()
        self.pattern_searcher.key = key  # attach the key vgroup to the pattern vgroup for later reference

        key.move_to(pattern_loc + (0,))
        key.width = 1.3
        key.height = min(key.height, 1.3)

        self.scene.play(Write(self.pattern_searcher), Write(key), run_time=0.2)

    def _move_pattern_searcher(self, top_left: Tuple[int, int], rate):
        # top left is the grid box that the top left of the pattern is in
        top_left = top_left[0]+1, top_left[1]+1  # add 1
        top_left_cell = self.table.get_cell(top_left)

        self.pattern_searcher.generate_target(True)
        self.pattern_searcher.target.set_x(top_left_cell.get_left()[0] + self.pattern_searcher.width/2)
        self.pattern_searcher.target.set_y(top_left_cell.get_top()[1] - self.pattern_searcher.height/2)

        self.scene.play(
            MoveToTarget(self.pattern_searcher, run_time=0.25*rate, rate_func=lambda x: x)  # linear
        )

    @property
    def road_mobjects(self):
        return filter(lambda x: x is not None, list(map(get_mob_or_none, np.array(self.matrix).flatten())))

    def add_road_block(self, road: Road, grid_pos):
        grid_pos = np.array(grid_pos)

        try:
            self.matrix[grid_pos[0]][grid_pos[1]] = road
        except IndexError:
            return
        cell = self.table.get_highlighted_cell(grid_pos[::-1] + 1)

        road.mob.width = cell.width
        road.mob.height = cell.height
        road.mob.set_x(cell.get_x())
        road.mob.set_y(cell.get_y())

        road.mob.set_z_index(1)  # keep road in front

        self.scene.add(road.mob)
        self.contained_mobjects.append(road.mob)

    def remove_road_block(self, grid_pos):
        grid_pos = np.array(grid_pos)

        road = self.matrix[grid_pos[0]][grid_pos[1]]

        try:
            self.contained_mobjects.remove(road.mob)
            self.scene.remove(road.mob)
        except AttributeError:
            return False

        self.matrix[grid_pos[0]][grid_pos[1]] = None
        return True

    def generate_roads(self, matrix, animate, pattern_loc):
        # pattern_loc is location of 'key' pattern
        generate_roads_from_patterns(self, matrix, animate, pattern_loc)
        if animate:  # remove pattern searcher
            self.scene.play(
                AnimationGroup(
                    *(FadeOut(obj) for obj in self.pattern_searcher.submobjects)
                ),
                AnimationGroup(
                    *(FadeOut(obj) for obj in self.pattern_searcher.key.submobjects)
                )
            )
        self.pattern_searcher = None

    def objects_excluding_road_section(self, section):
        mob_section_flat = list(map(get_mob_or_none, section.flatten()))
        contained_excluding_section = [obj for obj in self.contained_mobjects if obj not in mob_section_flat]
        return contained_excluding_section

    def focus_on_subgrid(self, tl_pos, br_pos, focus_loc=(0, 0), max_size=5):
        tl_pos, br_pos = np.array(tl_pos), np.array(br_pos)

        tl_manim_pos = self.table.get_cell(tl_pos+1).get_corner(UP+LEFT)

        section = np.array(cut_matrix(self.matrix, tl_pos, br_pos+1))

        contained_excluding_section = self.objects_excluding_road_section(section)

        self.scene.play(
            AnimationGroup(
                *(FadeOut(obj) for obj in contained_excluding_section)
            ),
            FadeOut(self.table)
        )

        [self.scene.remove(obj) for obj in contained_excluding_section]  # remove all

        # replace None with black square and flatten
        mobject_array = []
        for row in section:
            for col in row:
                if col is None:
                    new_mobj = Square(side_length=self.grid_size, color=BLACK)
                else:
                    new_mobj = col.mob

                mobject_array.append(new_mobj)

        subgrid_group = Group(
            *mobject_array
        )

        rows = abs(br_pos[0]-tl_pos[0])+1
        cols = abs(br_pos[1]-tl_pos[1])+1

        subgrid_group.arrange_in_grid(
            rows=rows,
            cols=cols,
            buff=0,
            flow_order="dr")

        subgrid_group.width = cols*self.grid_size

        subgrid_group.move_to(tl_manim_pos + np.array([subgrid_group.width / 2, -subgrid_group.height / 2, 0]))

        self.scene.add(subgrid_group)

        subgrid_group.generate_target(True)
        subgrid_group.target.move_to(focus_loc + (0,))
        subgrid_group.target.width = max_size
        subgrid_group.target.height = min(subgrid_group.target.height, max_size)

        self.scene.play(
            MoveToTarget(subgrid_group)
        )

        return subgrid_group

    def unfocus_subgrid(self, subgrid: Group, original_tl, original_br, animate_in=True):
        # redraw table, and other contained objects

        original_tl, original_br = np.array(original_tl), np.array(original_br)

        grid_width = original_br[1] - original_tl[1] + 1

        tl_manim_pos = self.table.get_cell(original_tl+1).get_corner(UP + LEFT)

        subgrid.generate_target(True)
        subgrid.target.width = grid_width*self.grid_size
        subgrid.target.move_to(tl_manim_pos + np.array([subgrid.target.width/2, -subgrid.target.height/2, 0]))

        print(subgrid.submobjects[0].get_fill_opacity())

        self.scene.play(
            MoveToTarget(subgrid)
        )

        section = np.array(cut_matrix(self.matrix, original_tl, original_br + 1))
        contained_excluding_section = self.objects_excluding_road_section(section)

        if animate_in:
            self.scene.play(
                Write(self.table),
                *(FadeIn(obj) for obj in contained_excluding_section)
            )

    def move_to(self, position, animate=True):
        position = np.array(position)
        difference = position - np.array(self.table.get_center()[:2])

        # items to move is table, and list of contained mobjects
        all_mobjects = [self.table] + self.contained_mobjects

        if animate:
            for obj in all_mobjects:
                obj.generate_target(True)
                obj.target.shift(tuple(difference) + (0,))

            self.scene.play(
                *(MoveToTarget(obj) for obj in all_mobjects),
                run_time=1.5
            )

        else:
            for obj in all_mobjects:
                obj.shift(tuple(difference) + (0,))
