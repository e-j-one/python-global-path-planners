import numpy as np
import pytest

from path_planners.rrt_utils.rrt_trees import RrtTree
from path_planners.rrt_utils.rrt_nodes import RrtNode


def test_rrt_tree():
    # Arrange
    root_pose = (1.0, 2.0, 0.3)
    node_1_pose = (4.0, 5.0, 0.6)
    node_2_pose = (-700.0, -800.0, -0.9)
    goal_pose = (10.0, 11.0, 1.2)

    num_nodes = 4

    root_pos = (root_pose[0], root_pose[1])
    node_1_pos = (node_1_pose[0], node_1_pose[1])
    node_2_pos = (node_2_pose[0], node_2_pose[1])

    pos_origin = (0.0, 0.0)
    pos_near_root = (1.1, 2.1)
    pos_near_node_1 = (4.1, 5.1)
    pos_near_node_2 = (-700.1, -800.1)
    pos_with_no_node = (1.0, 5.0)

    # Act
    tree = RrtTree()
    tree.add_root(root_pose)
    tree.add_node(node_1_pose, 0)
    tree.add_node(node_2_pose, 0)
    tree.add_node(goal_pose, 1)

    answer_num_nodes = tree.get_num_nodes()
    answer_pose_of_root = tree.get_pose_of_node(0)
    answer_pose_of_node_1 = tree.get_pose_of_node(1)
    answer_pose_of_node_2 = tree.get_pose_of_node(2)

    answer_idx_nearest_from_root_pose = tree.find_nearest_node(root_pos).get_idx()
    answer_idx_nearest_from_node_1_pose = tree.find_nearest_node(node_1_pos).get_idx()
    answer_idx_nearest_from_node_2_pose = tree.find_nearest_node(node_2_pos).get_idx()
    answer_idx_nearest_from_origin = tree.find_nearest_node(pos_origin).get_idx()
    answer_idx_nearest_pos_0 = tree.find_nearest_node(pos_near_root).get_idx()
    answer_idx_nearest_pos_1 = tree.find_nearest_node(pos_near_node_1).get_idx()
    answer_idx_nearest_pos_2 = tree.find_nearest_node(pos_near_node_2).get_idx()

    answer_root_in_tree = tree.check_if_pose_in_tree(root_pose)
    answer_node_1_in_tree = tree.check_if_pose_in_tree(node_1_pose)
    answer_node_2_in_tree = tree.check_if_pose_in_tree(node_2_pose)
    answer_pos_origin_in_tree = tree.check_if_pose_in_tree(pos_origin)
    answer_pos_near_root_in_tree = tree.check_if_pose_in_tree(pos_near_root)
    answer_pos_near_node_1_in_tree = tree.check_if_pose_in_tree(pos_near_node_1)
    answer_pos_near_node_2_in_tree = tree.check_if_pose_in_tree(pos_near_node_2)
    answer_pos_with_no_node_in_tree = tree.check_if_pose_in_tree(pos_with_no_node)

    answer_idx_of_root = tree.get_idx_from_pos(root_pose)
    answer_idx_of_node_1 = tree.get_idx_from_pos(node_1_pose)
    answer_idx_of_node_2 = tree.get_idx_from_pos(node_2_pose)
    answer_idx_of_pos_near_root = tree.get_idx_from_pos(pos_near_root)
    answer_idx_of_pos_near_node_1 = tree.get_idx_from_pos(pos_near_node_1)
    answer_idx_of_pos_near_node_2 = tree.get_idx_from_pos(pos_near_node_2)
    answer_idx_of_pos_with_no_node = tree.get_idx_from_pos(pos_with_no_node)

    answer_path_from_goal = tree.get_path_from_tree(goal_pose)
    answer_path_from_pos_with_no_node = tree.get_path_from_tree(pos_with_no_node)

    answer_path_idx_from_goal = tree.get_path_idx_from_tree(goal_pose)
    answer_path_idx_from_pos_with_no_node = tree.get_path_idx_from_tree(
        pos_with_no_node
    )

    tree.reset_tree()
    answer_num_nodes_after_reset = tree.get_num_nodes()

    # Assert
    assert answer_num_nodes == num_nodes
    assert answer_pose_of_root == root_pose
    assert answer_pose_of_node_1 == node_1_pose
    assert answer_pose_of_node_2 == node_2_pose

    # Test find_nearest_node method
    assert answer_idx_nearest_from_root_pose == 0
    assert answer_idx_nearest_from_node_1_pose == 1
    assert answer_idx_nearest_from_node_2_pose == 2
    assert answer_idx_nearest_from_origin == 0
    assert answer_idx_nearest_pos_0 == 0
    assert answer_idx_nearest_pos_1 == 1
    assert answer_idx_nearest_pos_2 == 2

    # Test check_if_pose_in_tree method
    assert answer_root_in_tree == True
    assert answer_node_1_in_tree == True
    assert answer_node_2_in_tree == True
    assert answer_pos_origin_in_tree == False
    assert answer_pos_near_root_in_tree == False
    assert answer_pos_near_node_1_in_tree == False
    assert answer_pos_near_node_2_in_tree == False
    assert answer_pos_with_no_node_in_tree == False

    # Test get_idx_from_pos method
    assert answer_idx_of_root == 0
    assert answer_idx_of_node_1 == 1
    assert answer_idx_of_node_2 == 2
    assert answer_idx_of_pos_near_root == -1
    assert answer_idx_of_pos_near_node_1 == -1
    assert answer_idx_of_pos_near_node_2 == -1
    assert answer_idx_of_pos_with_no_node == -1

    # Test get_path_from_tree method
    assert answer_path_from_goal == [root_pose, node_1_pose, goal_pose]
    assert answer_path_from_pos_with_no_node == None

    # Test get_path_idx_from_tree method
    assert answer_path_idx_from_goal == [0, 1, 3]
    assert answer_path_idx_from_pos_with_no_node == None

    # Test reset_tree method
    assert answer_num_nodes_after_reset == 0

    # Test add_root twice
    with pytest.raises(ValueError, match=r".*Root node already exists*"):
        tree.add_root(root_pose)
        tree.add_root(root_pose)
