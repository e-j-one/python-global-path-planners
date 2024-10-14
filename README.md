# Python Global Path Planners

## Implemented Algorithms

### RRT*
For unicycle robot.

Until goal is reached or max_iter is reached:
1. Sample a random point
2. Find the nearest node (by position) in the tree to the random point
3. Drive from the nearest node towards the random point and get new node position
4. Find the nodes withing near_node_threshold distance from the new node position
5. Choose the parent node with minimum cost from the near nodes
    - For each near node:
        - Get v (const), w from the near node to the new node
        - Calculate the cost
        - If cost is smaller than minimum cost,
          check collision between near node and new node.
          If no collision, update the minimum cost and parent node
6. Add the new node to the tree with the parent node
    - The heading of the new node is assigned by the path from the parent node chosen for that node.
7. Rewire the tree
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
