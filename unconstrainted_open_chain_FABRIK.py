import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_test_position():
    """
        Set the chain in initial vertical position.
        Output:
            chain - Nx3 numpy array of nodes position in space.
                chain[0] - is the root node by default.
    """
    # Links lengths.
    link_lens = np.array([1., 1., 1., 1.])

    joints_num = len(link_lens) + 1

    # We want our chain to be vertical extended, so unit_vector will be the
    # direction of chain.
    unit_vector = np.array([0., 0., 1.])

    # Initialize all nodes positions.
    chain = np.repeat([unit_vector], joints_num, axis=0)
    curr_node_height = 0.
    for i in xrange(joints_num):
        chain[i] *= curr_node_height
        if i != joints_num - 1:
            curr_node_height += link_lens[i]
    return chain


def get_links_lengths(chain):
    """
        Check links lengths.
        Input:
            chain - Mx3 numpy array of joints 3D positions.
        Output:
            links_lengths - (M-1) array of links lengths.
    """
    links_num = chain.shape[0] - 1
    links_lengths = np.zeros(links_num)
    for i in xrange(links_num):
        links_lengths[i] = np.linalg.norm(chain[i] - chain[i + 1])
    return links_lengths


def FABRIK_open_chain_solver(init_chain_position, target_end_effector_pos):
    """
        FABRIK IK solver.
        Input:
            init_chain_position - (Mx3 numpy array) initial postion of kinematic chain.
            target_end_effector_pos - (numpy array of size 3) the 3D target position of the end effector.
        Output:
            is_reached - (True/False) is the end effector in the target position.
            current_chain_position - (Mx3 numpy array) the current position of kinematic chain.
        For details and pseudocode see part 4 in Ref:
            FABRIK: A fast, iterative solver for the Inverse Kinematics problem
            By: Aristidou, Andreas; Lasenby, Joan
            GRAPHICAL MODELS  Volume: 73   Pages: 243-260   Published: 2011
    """
    # Constants.
    tolerance = 0.01
    max_iter_num = 100

    is_reached = False
    link_lens = get_links_lengths(init_chain_position)
    chain_total_length = link_lens.sum()

    # Check if reachable.
    if np.linalg.norm(init_chain_position[0] - target_end_effector_pos) <= chain_total_length:
        init_root_position = init_chain_position[0]
        current_chain_position = init_chain_position.copy()

        dist = np.linalg.norm(init_chain_position[-1] - target_end_effector_pos)
        iter_num = 0
        # Do until the target position reached or maximum iterations number is reached.
        while (dist > tolerance) and (iter_num < max_iter_num):
            # STAGE 1: FORWARD REACHING
            current_chain_position[-1] = target_end_effector_pos
            for i in reversed(range(current_chain_position.shape[0] - 1)):
                current_link_len = np.linalg.norm(current_chain_position[i] - current_chain_position[i + 1])
                lambda_v = link_lens[i] / current_link_len
                current_chain_position[i] = (1. - lambda_v) * current_chain_position[i + 1] + lambda_v * current_chain_position[i]

            # STAGE 2: BACKWARD REACHING
            current_chain_position[0] = init_root_position
            for i in range(current_chain_position.shape[0] - 1):
                current_link_len = np.linalg.norm(current_chain_position[i] - current_chain_position[i + 1])
                lambda_v = link_lens[i] / current_link_len
                current_chain_position[i + 1] = (1. - lambda_v) * current_chain_position[i] + lambda_v * current_chain_position[i + 1]

            dist = np.linalg.norm(current_chain_position[-1] - target_end_effector_pos)
            iter_num += 1

        # Check if the targed position is reached.
        if dist < tolerance:
            is_reached = True
    else:
        raise ValueError('The target is unreachable:\n\
                         Maximum length of chain = %f\n\
                         Euqlidean distance between root joint and target = %f'
                         % (chain_total_length, np.linalg.norm(init_chain_position[0] - target_end_effector_pos)))

    return is_reached, current_chain_position


def plot_FABRIK_solution(target_end_effector_pos, chain_position_list):
    """
        Testing plot.
        Input:
            target_end_effector_pos - (numpy array 3D) the target position of end effector.
            chain_position_list - (list of Mx3 numpy array) list of kinematic chain position.
        Comments:
            Kinematic chains could appeal as they have different links lengths, because axis unit lengths couldn't be equal.
            This is a Matplotlib bug and there is some workaround for it, but it doesn't work complitly.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', projection='3d')
    for ind, position in enumerate(chain_position_list):
        ax.plot(position[:, 0], position[:, 1], position[:, 2], marker='o', label=str(ind))
        if ind > 0:
            ax.plot([target_end_effector_pos[ind - 1][0]], [target_end_effector_pos[ind - 1][1]], [target_end_effector_pos[ind - 1][2]], marker='o', linestyle='.', color='r')
    ax.legend(loc=0, numpoints=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Magic workaround for the Matplolib bug.
    coub_axis = list()
    for ind in xrange(3):
        coub_axis.append(np.array(map(lambda x: x[:, ind].max(), chain_position_list)
                                  + map(lambda x: x[:, ind].min(), chain_position_list)
                                  + map(lambda x: x[ind].max(), target_end_effector_pos)
                                  + map(lambda x: x[ind].max(), target_end_effector_pos)))

    #                            + [target_end_effector_pos[ind]])
    # Create cubic bounding box to simulate equal aspect ratio
    # max_range = np.array([coub_axis[0].max() - coub_axis[0].min(), coub_axis[1].max() - coub_axis[1].min(), coub_axis[2].max() - coub_axis[2].min()]).max()
    max_range = np.array(map(lambda axis: axis.max() - axis.min(), coub_axis)).max()
    grid_list = list()
    for ind in xrange(3):
        curr_axis_grid = 0.5 * max_range * np.mgrid[-1: 2: 2, -1: 2: 2, -1: 2: 2][ind].flatten() + 0.5 * (coub_axis[ind].max() + coub_axis[ind].min())
        grid_list.append(curr_axis_grid)

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(grid_list[0], grid_list[1], grid_list[2]):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()
    pass


def main():
    # Set the initial position of kinematic chain.
    init_chain_position = set_test_position()

    # Set the list of target positions of the end effector.
    target_end_effector_positions = [np.array([1., 3., 1.]),
                                     np.array([3., 1., 0.]),
                                     np.array([-3., -1., 0.])]

    # Solve all chain positions.
    positions_list = [init_chain_position.copy()]
    res_chain_position = init_chain_position.copy()
    for next_target in target_end_effector_positions:
        is_reached, res_chain_position = FABRIK_open_chain_solver(res_chain_position, next_target)
        if is_reached:
            positions_list.append(res_chain_position.copy())
        else:
            print 'next target pos:', next_target, ' is not reachable!'
            break

    # Check if links lengths are the same as initial.
    print 'Control links lengths: ', get_links_lengths(res_chain_position)

    # Plot solution.
    plot_FABRIK_solution(target_end_effector_positions, positions_list)
    pass


if __name__ == "__main__":
    main()
    pass
