# [WIP] Python Global Path Planners

## Implemented Algorithms

### RRT-unicycle
Until goal is reached or max_iter is reached:
1. Sample a random point (x, y).
2. Find the nearest node (by position) in the tree to the random point.
3. Drive from the nearest node towards the random point and get new node position.
4. Check collision and reachability.
    - If collision or unreachable, continue.
5. Add the new node to the tree with the parent node.
    - The heading of the new node is assigned by the path from the parent node chosen for that node.

### RRT*
For unicycle robot.

<p align="center">
  <img width="40%" src="docs/figures/rrt_star_smooth_unicycle_tree.png" align="center" alt="fig1" />
  <img width="40%" src="docs/figures/rrt_star_smooth_unicycle_path.png" align="center" alt="fig2" />
  <figcaption align="center">Fig. 1: RRT*-SmoothUnicycle tree and path with path length as a cost.</figcaption>
</p>


1. Sample a random point (x, y).
2. Find the nearest node (by position) in the tree to the random point.
3. Drive from the nearest node towards the random point and get new node position.
4. Find the nodes withing near_node_threshold distance from the new node position.
5. Choose the parent node with minimum cost from the near nodes.
    - For each near node:
        - Get arc path from the near node to the new node.
        - Check collision and calculate the cost of the path.
6. Add the new node to the tree with the parent node.
    - The heading of the new node is assigned by the path from the parent node chosen for that node.
7. Rewire the tree.
    - To check if two pose can be connected, try connecting poses only using the two arc.
    - If the pose connecting two arc (the intersection pose) is within dist_near from both poses, they can be rewired.
    - If rewired, the pose connecting two arcs is also added as a node.

#### Difference from RRT*-Unicycle (Yokoyama, 2023)
- Robot does not pivot.
- In rewiring step, dubins path is used. 
    TODO: add graceful control (park, 2011)

## How to run tests.
```bash
bash run_tests.sh
```

# TODO
Rename check_unicycle_reachability -> check_unicycle_pos_reachability
# References
[RRT*-unicycle](https://github.com/naokiyokoyama/rrt_star)
