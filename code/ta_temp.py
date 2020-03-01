# temp task assignment 

class AA_Params:
    """
    Parameters for the auction
    """
    def __init__(self, max_comm_iter = None, R_comm = None, N = None):
        """
        Params:
            max_comm_iter (int): Maximum number of iterations in the auction
            N (int): The number of drones
            R_comm (float): The radius of communication. Drones
                can only communicate when they are within R_comm
                of each other.
        """
        self.max_comm_iter = max_comm_iter
        self.R_comm = R_comm
        self.N = N

import numpy as np
from auction.bid import bid
import lib.config as config
from a_star.a_star import get_all_a_star_trajectory

def auction(current_pos, xf, m, p, j, AA_param, verbose = 0, use_a_star_distance=0, walls=[], MPC_param=None):
    """
    Holds an auction to determine which drone should be given each target location. I.e.
    maps current_pos to xf for each drone. The distance metric can either be euclidean
    distance or A* shortest path distance.
    Params:
        current_pos (numpy array): 2xN array of the current positions of the drones
        xf (numpy array): 2xN array of the final positions of the drones
        m (numpy array): N array of integers, representing number of 
            target locations to bid on.
        p (numpy array): N x M array of bid estimates.
        j (numpy array): N array of current assignments to positions.
            The i'th drone is assigned to the location at index j[i]
        AA_param (Custom object): Configuration parameters for the auction
        verbose (int): How much to log to the console (min -1, max 3)
            The lower the number, the less to log. -1 means don't log
            anything (including warnings/errors). 3 means log everything,
            including print outs of most calculated arrays.
        use_a_star_distance (bool): If true, then use min path distance for the
            auction. Otherwise, use the euclidean distance
            It is HIGHLY recommended to use the Euclidean distance to save time.
        walls (numpy array): Wx2x2 numpy array of the (x_i, y_i), (x_f, y_f) positions
            of any walls (or other obstacles)
            Necessary to get min-path distance
        MPC_param (Custom object): Parameters for the SCP/MPC
    """
    N             = AA_param.N
    R_comm        = AA_param.R_comm
    max_comm_iter = AA_param.max_comm_iter

    M = xf.shape[1]

    assert N == M, "Currently, differing numbers of start and end positions are not supported"

    if p is None:
        p     = np.zeros((N, M))
        p_old = -np.ones((N, M))
    else:
        p_old = p

    if j is None:
        j = np.ones((N), dtype=np.int64)

    p_temp = np.zeros((N, M))
    done   = np.zeros((N))
    count  = np.zeros((N))

    # Get the costs of all trajectories
    if use_a_star_distance:
        if verbose > 0:
            print("using a star distance")
        padded_walls = []
        for wall in walls:
            padded_walls.append(add_padding(wall, padding=0.05))

        padded_walls = np.asarray(padded_walls)
        resolution = .05 # how fine is our grid; distance between cols
        grid_size = MPC_param.pos_space * 2

        # 2nd parameter is costs of the trajectories
        _, c = get_all_a_star_trajectory(grid_size, resolution, padded_walls, 
                                         current_pos, xf, MPC_param.T, MPC_param.dt, verbose)
    else:
        c = get_euclidean_distance(N, M, current_pos, xf)

    if verbose > 1:
        print("c:\n", c)
        print("\n\ncurrent_pos:\n", current_pos)
        print("\n\nxf:\n", xf)

    # Until we have a satisfying solution, repeatedly have all members bid
    # on locations.
    while not np.all(done):
        # Bid on locations
        for i in range(N):
            if not done[i]:
                p[i, :], j[i], m[i], count[i] = \
                    bid(c[i, :], p[i, :], p_old[i, :], j[i], m[i], count[i])
                p_old[i, :] = p[i, :]
                done[i] = count[i] > max_comm_iter

        # Communicate with all neighbors in range
        for i in range(N):
            neighbors = np.where(np.sqrt(
                                    np.square((current_pos[0, i] - current_pos[0, :])) +
                                    np.square((current_pos[1, i] - current_pos[1, :]))
                                 ) <= R_comm)[0]
            p_comm = p[neighbors]

            if np.any(p_comm < 0):
                for k in range(p_comm.shape[1]):
                    max_val = np.amax(abs(p_comm[:, k]))
                    max_indices = np.where(abs(p_comm[:, k]) == max_val)
                    p_temp[i, k] = max_val * (2 * np.all(np.sign(p_comm[max_indices, k]) == 1) - 1)

            else:
                p_temp[i, :] = np.amax(abs(p_comm), axis=0)

        p = p_temp

    # FIXME: Why are there duplicates???
    # This code is designed to get rid of cases wherein the same drone gets assigned
    # to two separate locations; it seems to be an issue with having differing numbers
    # of starting and ending locations, although the exact cause is to be determined.
    if len(np.unique(j)) < len(j) and verbose > -1:
        print("WARNING: Duplicate allocations found in auction. Resolving automatically.")

        while len(np.unique(j)) < len(j):
            not_used = np.setdiff1d(np.arange(j.shape[0]), np.unique(j))
            vals, indices, counts = np.unique(j, return_counts=True, return_index=True)
            duplicate_indices = np.where(counts[:] > 1)
            j[indices[duplicate_indices]] = np.random.choice(not_used, size=j[duplicate_indices].shape)

    final_pos = xf[:, j]

    return final_pos, m, p, j

def get_euclidean_distance(N, M, current_pos, xf):
    """
    Euclidean distance from each drone to each final point.
    Return:
        the cost array (NxM numpy array)
    """
    c      = np.zeros((N, M))
    for i in range(N):
        for k in range(M):
            c[i,k] = np.linalg.norm(current_pos[:,i] - xf[:, k])

    return c


def add_padding(iterable, padding=0.075):
    """
    Adds a little bit of padding to the walls to make sure there's no
    issues due to discretization
    """
    if iterable.size == 0:
        return np.array([])

    min_x, min_y = np.min(iterable, axis=0)
    max_x, max_y = np.max(iterable, axis=0)

    return np.array([[min_x - padding, min_y - padding], [max_x + padding, max_y + padding]])

import numpy as np
import lib.config as config

def bid(c, p, p_old, j, m_j, count, verbose = 0):
    """
    Simulate one agent (j's) bidding process
    Params:
        c (numpy array): NxM array of floating point costs for getting
            drone n to position m.
        p (numpy array): N x M array of bid estimates.
        p_old (numpy array): N x M array of last round's bid estimates.
        j (integer): Current index under consideration
        m_j (integer): number of target locations to bid on.
        count (integer): Used to keep track of how long the bidding
            has going on; stopping the bidding at count=2 * diameter of the net.
        verbose (int): How much to log to the console (min -1, max 3)
            The lower the number, the less to log. -1 means don't log
            anything (including warnings/errors). 3 means log everything,
            including print outs of most calculated arrays.
    """
    eps = 0.001

    # Agent j is outbid
    if abs(p[j]) > p_old[j]:
        m_j = max(m_j, len(np.nonzero(abs(p)>0)[0]))

        if len(np.where(p>0)[0]) >= m_j:
            # FIXME: This didn't use to be a min
            # As written, we are unable to support dynamically growing
            # the number of targets.
            m_j = min(len(p), len(np.nonzero(p>0)[0]) + 1)
            p[:m_j] = - (abs(p[:m_j]) + eps)

        j = np.argmin(abs(p[:m_j]) + c[:m_j])
        v = np.amin(abs(p[:m_j]) + c[:m_j])
        relevant_indices = np.concatenate((np.arange(j), np.arange(j+1, m_j)))
        w = np.amin(abs(p[relevant_indices]) + c[relevant_indices])
        gamma = w - v + eps
        p[j] = abs(p[j]) + gamma

        count = 0

    # Another agent is outbid
    elif np.any(p != p_old):
        count = 0
        m_j = max(m_j, len(np.nonzero(abs(p)>0)[0]))
    else:
        count += 1

    return p, j, m_j, count    