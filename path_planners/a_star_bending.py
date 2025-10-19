from typing import Tuple, Optional, List
import time
import heapq

import numpy as np

import path_planners.utils.gridmap_utils as GridmapUtils

from path_planners.a_star import AStarPlanner


class AStarBendingPlanner(AStarPlanner):
    """
    Plan a path from start to goal using A* algorithm.
    """

    def __init__(
        self,
        terminate_on_goal_reached: bool = True,
        goal_reach_dist_threshold: float = 0.5,
        goal_reach_angle_threshold: float = 0.1 * np.pi,
        occupancy_map_obstacle_padding_dist: float = 0.5,
        interpolate_path: bool = False,
        d_s: float = 0.25,
        collision_check_ratio_to_map_res: float = 0.8,
        print_log: bool = False,
    ):
        super().__init__(
            terminate_on_goal_reached,
            goal_reach_dist_threshold,
            goal_reach_angle_threshold,
            occupancy_map_obstacle_padding_dist,
            interpolate_path,
            d_s,
            collision_check_ratio_to_map_res,
            print_log,
        )

    def plan_global_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        render=False,
        save_plot_to_file=False,
        plot_file_name: str = "a_star_bending_path.png",
    ) -> Tuple[bool, List[Tuple[float, float, float]], int]:
        return super().plan_global_path(
            start_pose, goal_pose, render, save_plot_to_file, plot_file_name
        )

    def _plan_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], int]:
        """
        Returns
        -------
        path : Optional[List[Tuple[float, float, float]]]
            The planned path. None if the path is not found.
        num_nodes_sampled : int
            The number of nodes sampled during the planning.
            -1 if the path planner is not sampling based.
        """

        num_nodes_sampled = -1  # since this is not a sampling based planner

        path_planning_start_time = time.time()

        # Get start and goal grid cell idx
        start_cell = GridmapUtils.pos_to_grid_cell_idx(
            start_pose, self._occupancy_map_resolution, self._occupancy_map_origin
        )
        goal_cell = GridmapUtils.pos_to_grid_cell_idx(
            goal_pose, self._occupancy_map_resolution, self._occupancy_map_origin
        )
        self._goal_cell = goal_cell

        # Check if the start and goal cells are valid
        if not GridmapUtils.check_if_cell_is_free(self._occupancy_map, start_cell):
            Warning("Start cell is not free!")
            return None, num_nodes_sampled
        if not GridmapUtils.check_if_cell_is_free(self._occupancy_map, goal_cell):
            Warning("Goal cell is not free!")
            return None, num_nodes_sampled

        self._g_score = {start_cell: 0}  # g_score[cell] = cost to reach cell
        self._parent = {start_cell: start_cell}  # parent[cell] = parent cell
        self._open_set = []
        self._open_set_tracker = {}

        heapq.heappush(self._open_set, (self._heuristic_score(start_cell), start_cell))
        self._open_set_tracker[start_cell] = self._heuristic_score(start_cell)

        # closed_set = set()

        i = 0

        while len(self._open_set) > 0:
            _, current_cell = heapq.heappop(self._open_set)
            self._open_set_tracker.pop(current_cell)
            i += 1

            if current_cell == goal_cell:
                if self._print_log:
                    print("Goal is reached!!!")

                cell_path = self._reconstruct_path(self._parent, current_cell)
                path = self._get_pose_path_from_cell_path(cell_path)

                path_planning_end_time = time.time()
                if self._print_log:
                    print(
                        f"Path planning time: {path_planning_end_time - path_planning_start_time:.2f} sec"
                    )
                return path, num_nodes_sampled

            # closed_set.add(current_cell)
            for neighbor in self._get_neighbors(current_cell):
                # if neighbor in closed_set:
                #     continue

                tentative_g_score = self._g_score[current_cell] + self._get_cost(
                    current_cell, neighbor
                )

                if tentative_g_score < self._g_score.get(neighbor, float("inf")):
                    self._parent[neighbor] = current_cell
                    self._g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic_score(neighbor)

                    if neighbor not in self._open_set_tracker:
                        heapq.heappush(self._open_set, (f_score, neighbor))
                        self._open_set_tracker[neighbor] = f_score

                # if neighbor not in self._open_set:
                #     self._g_score[neighbor] = float("inf")
                #     self._parent[neighbor] = None
                # self._update_vertex(current_cell, neighbor)

            if self._print_log and i % 1000 == 0:
                print(f"len(open_set): {len(self._open_set)}")

        path_planning_end_time = time.time()
        if self._print_log:
            print("Goal is not reachable!")
            print(
                f"Path planning time: {path_planning_end_time - path_planning_start_time:.2f} sec"
            )

        return None, num_nodes_sampled

    def _get_neighbors(self, current_cell: Tuple[int, int]):
        """
        Get the neighbors of the current cell.
        """
        neighbors = []
        for i, j in [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
        ]:
            neighbor = (current_cell[0] + i, current_cell[1] + j)
            if GridmapUtils.check_if_cell_is_free(self._padded_occupancy_map, neighbor):
                neighbors.append(neighbor)
        return neighbors

    def _get_cost(self, cell_i, cell_f):
        parent = self._parent.get(cell_i, None)
        cost = np.linalg.norm(np.array(cell_i) - np.array(cell_f))

        if parent == cell_f:
            cost += 2.0  # bending penalty
        if not (parent[0] == cell_i[0] and cell_i[0] == cell_f[0]) and not (
            parent[1] == cell_i[1] and cell_i[1] == cell_f[1]
        ):
            cost += 1.0  # bending penalty
        return cost
