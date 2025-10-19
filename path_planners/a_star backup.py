from typing import Tuple, Optional, List, Set
import time
import heapq
import math

import numpy as np

import path_planners.utils.gridmap_utils as GridmapUtils

from path_planners.path_planner import PathPlanner


class AStarPlanner(PathPlanner):
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
        plot_file_name: str = "a_star_path.png",
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
        A* path planner (no Theta* line-of-sight).
        Robust features:
        - Stale-heap entry skipping for decrease-key emulation
        - Reopen nodes in closed_set if we discover a better g
        - Uses f = g + h consistently
        """
        num_nodes_sampled = -1  # not a sampling-based planner
        t0 = time.time()

        # ---- helpers -----------------------------------------------------------
        def step_cost(u, v) -> float:
            # Use the same metric as your heuristic!
            # Default: 8-connected Euclidean cost scaled by map resolution.
            dx = v[0] - u[0]
            dy = v[1] - u[1]
            return math.hypot(dx, dy) * self._occupancy_map_resolution

        # ---- grid indices for start/goal --------------------------------------
        start_cell = GridmapUtils.pos_to_grid_cell_idx(
            start_pose, self._occupancy_map_resolution, self._occupancy_map_origin
        )
        goal_cell = GridmapUtils.pos_to_grid_cell_idx(
            goal_pose, self._occupancy_map_resolution, self._occupancy_map_origin
        )
        self._goal_cell = goal_cell  # assuming your heuristic reads this

        # ---- validity checks ---------------------------------------------------
        if not GridmapUtils.check_if_cell_is_free(self._occupancy_map, start_cell):
            Warning("Start cell is not free!")
            return None, num_nodes_sampled
        if not GridmapUtils.check_if_cell_is_free(self._occupancy_map, goal_cell):
            Warning("Goal cell is not free!")
            return None, num_nodes_sampled

        # ---- state -------------------------------------------------------------
        g = {start_cell: 0.0}
        parent = {start_cell: start_cell}

        # min-heap of (f, cell); tracker maps cell -> current best f
        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        open_f = {}

        f_start = self._heuristic_score(start_cell)  # since g(start)=0
        heapq.heappush(open_heap, (f_start, start_cell))
        open_f[start_cell] = f_start

        closed_set: Set[Tuple[int, int]] = set()

        # ---- main loop ---------------------------------------------------------
        while open_heap:
            f_pop, current = heapq.heappop(open_heap)

            # Skip stale entries (older, worse f for the same node)
            if current not in open_f or f_pop != open_f[current]:
                continue

            # We are expanding 'current' now; remove from tracker
            open_f.pop(current, None)

            if current == goal_cell:
                if self._print_log:
                    print("Goal is reached!!!")
                cell_path = self._reconstruct_path(parent, current)
                path = self._get_pose_path_from_cell_path(cell_path)
                if self._print_log:
                    print(f"Path planning time: {time.time() - t0:.2f} sec")
                return path, num_nodes_sampled

            closed_set.add(current)

            for nbr in self._get_neighbors(current):
                # Skip blocked neighbors early
                if not GridmapUtils.check_if_cell_is_free(self._occupancy_map, nbr):
                    continue

                tentative_g = g[current] + step_cost(current, nbr)
                old_g = g.get(nbr, float("inf"))

                if tentative_g < old_g:
                    parent[nbr] = current
                    g[nbr] = tentative_g
                    f_nbr = tentative_g + self._heuristic_score(nbr)

                    # If nbr was previously closed with a worse g, "re-open" it
                    if nbr in closed_set:
                        closed_set.remove(nbr)

                    # Push new/better entry and record its best-known f
                    heapq.heappush(open_heap, (f_nbr, nbr))
                    open_f[nbr] = f_nbr

            if (
                self._print_log
                and (len(closed_set) % 1000 == 0)
                and len(closed_set) > 0
            ):
                print(
                    f"len(closed_set): {len(closed_set)}, len(open_set): {len(open_heap)}"
                )

        # ---- failure -----------------------------------------------------------
        if self._print_log:
            print("Goal is not reachable!")
            print(f"Path planning time: {time.time() - t0:.2f} sec")
        return None, num_nodes_sampled

    def _update_vertex(self, current_cell, neighbor):
        """
        Pure A* relaxation: only considers the edge (current_cell -> neighbor),
        """
        # Use the SAME edge/transition cost you used in Theta* for adjacent moves.
        # If you had something like self._edge_cost(u, v) or self._move_cost(u, v),
        # call it here. As a safe placeholder, this uses 8-connected Euclidean:
        dx = neighbor[0] - current_cell[0]
        dy = neighbor[1] - current_cell[1]
        step_cost = (
            math.hypot(dx, dy) * self._occupancy_map_resolution
        )  # replace if you have a dedicated cost fn

        tentative_g = self._g_score[current_cell] + step_cost
        if tentative_g < self._g_score.get(neighbor, float("inf")):
            # standard A*: parent is ALWAYS the current node on improvement
            self._parent[neighbor] = current_cell
            self._g_score[neighbor] = tentative_g

            f = tentative_g + self._heuristic_score(neighbor)  # f = g + h

            # "decrease-key" via push + tracker (heap has no native decrease-key)
            prev = self._open_set_tracker.get(neighbor, float("inf"))
            if f < prev:
                heapq.heappush(self._open_set, (f, neighbor))
                self._open_set_tracker[neighbor] = f

    def _heuristic_score(self, cell):
        """
        Returns the heuristic cost estimate from cell to goal_cell.
        """
        # return np.linalg.norm(np.array(cell) - np.array(self._goal_cell))
        return 0.0

    def _reconstruct_path(
        self, parent: dict, current_cell: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """
        Reconstruct the path from the parent dictionary.
        """
        path = [current_cell]
        while parent[current_cell] != current_cell:
            current_cell = parent[current_cell]
            path.append(current_cell)
        return path[::-1]

    def _get_pose_path_from_cell_path(self, cell_path: List[Tuple[int, int]]):
        return [
            (
                *GridmapUtils.grid_cell_idx_to_pos(
                    cell, self._occupancy_map_resolution, self._occupancy_map_origin
                ),
                0.0,
            )
            for cell in cell_path
        ]

    def _get_neighbors(self, current_cell: Tuple[int, int]):
        """
        Get the neighbors of the current cell.
        """
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor = (current_cell[0] + i, current_cell[1] + j)
                if GridmapUtils.check_if_cell_is_free(
                    self._padded_occupancy_map, neighbor
                ):
                    neighbors.append(neighbor)
        return neighbors

    def _get_cost(self, cell_i, cell_f):
        return np.linalg.norm(np.array(cell_i) - np.array(cell_f))
