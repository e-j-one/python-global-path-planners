from path_planners.rrt_utils.rrt_nodes import RrtNode, RrtStarNode


def test_rrt_node():
    # Arrange
    parent = 1
    idx = 2
    pose = (3.0, 4.0, 5.0)

    children = [6, 7, 8]

    # Act
    rrt_node = RrtNode(idx, pose, parent)
    for child in children:
        rrt_node.add_child(child)

    answer_idx = rrt_node.get_idx()
    answer_pos = rrt_node.get_pos()
    answer_pose = rrt_node.get_pose()
    answer_parent = rrt_node.get_parent()
    answer_children = rrt_node.get_children()

    # Assert
    assert answer_idx == idx
    assert answer_pos == (pose[0], pose[1])
    assert answer_pose == pose
    assert answer_parent == parent
    assert answer_children == children


def test_rrt_star_node():
    # Arrange
    parent = 1
    idx = 2
    pose = (3.0, 4.0, 5.0)

    children = [6, 7, 8]

    cost = 90.0
    cost_from_parent = 10.0

    updated_parent = 11
    updated_cost = 12.0
    updated_cost_from_parent = 1.3

    updated_updated_cost = 1.4

    # Act
    rrt_node = RrtStarNode(
        idx=idx, pose=pose, parent=parent, cost=cost, cost_from_parent=cost_from_parent
    )

    for child in children:
        rrt_node.add_child(child)

    answer_idx = rrt_node.get_idx()
    answer_pos = rrt_node.get_pos()
    answer_pose = rrt_node.get_pose()
    answer_parent = rrt_node.get_parent()
    answer_children = rrt_node.get_children()

    answer_cost = rrt_node.get_cost()
    answer_cost_from_parent = rrt_node.get_cost_from_parent()

    rrt_node.update_parent_and_cost(
        updated_parent, updated_cost, updated_cost_from_parent
    )

    answer_updated_parent = rrt_node.get_parent()
    answer_updated_cost = rrt_node.get_cost()
    answer_updated_cost_from_parent = rrt_node.get_cost_from_parent()

    rrt_node.update_cost(updated_updated_cost)

    answer_updated_updated_cost = rrt_node.get_cost()

    rrt_node.remove_child(children[0])
    answer_updated_children = rrt_node.get_children()

    # Assert
    assert answer_idx == idx
    assert answer_pos == (pose[0], pose[1])
    assert answer_pose == pose
    assert answer_parent == parent
    assert answer_children == children
    assert answer_cost == cost
    assert answer_cost_from_parent == cost_from_parent
    # Test update_parent_and_cost method
    assert answer_updated_parent == updated_parent
    assert answer_updated_cost == updated_cost
    assert answer_updated_cost_from_parent == updated_cost_from_parent
    # Test update_cost method
    assert answer_updated_updated_cost == updated_updated_cost
    # Test remove_child method
    assert children[0] not in answer_updated_children
