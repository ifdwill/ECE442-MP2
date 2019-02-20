# -*- coding: utf-8 -*-
import numpy as np

from collections import namedtuple, defaultdict

VarAssignment = namedtuple('VarAssignment', 'loc rot flip occupied')

class Assignment:
    def __init__(self, is_assigned, unassigned_vars, end_locations, variable_assignment):
        self.is_assigned = is_assigned
        self.unassigned_vars = unassigned_vars
        self.end_locations = end_locations
        self.variable_assignment = variable_assignment


class ConstraintProblem:
    def __init__(self, board, variables, valid_locations, occupied_locations, rotated_pents):
        self.board = board
        self.variables = variables
        self.valid_locations = valid_locations
        self.occupied_locations = occupied_locations
        self.rotated_pents = rotated_pents

# Assignment = namedtuple('Assignment', 'is_assigned unassigned_vars end_locations')
# ConstraintProblem = namedtuple('ConstraintProblem', 'board variables valid_locations occupied_locations')


def goal_test(assignment):
    return len(assignment.unassigned_vars) == 0


def select_unassigned_var(assignment, csp):
    if assignment.unassigned_vars == 0:
        return None

    # LRV
    var_dict = assignment.variable_assignment
    num_remaining = 0xffffffff
    min_remaining = []
    for key in assignment.unassigned_vars:
        if len(var_dict[key]) < num_remaining:
            num_remaining = len(var_dict[key])
            min_remaining.clear()
            min_remaining.append(key)

        elif len(var_dict[key]) == num_remaining:
            min_remaining.append(key)

    if len(min_remaining) == 1:
        return min_remaining[0]

    # MCV
    # for i in min_remaining:


    return min_remaining[0]


def old_ordered_domain_values(next_var, csp):
    pent = csp.variables[next_var]
    shape = pent.shape
    shape = (csp.board.shape[0] - shape[0] + 1, csp.board.shape[1] - shape[1] + 1)

    curr_pent = pent.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), curr_pent)

    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), np.flip(curr_pent))

    curr_pent = np.rot90(curr_pent)
    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), curr_pent)

    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), np.flip(curr_pent))

    curr_pent = np.rot90(curr_pent)
    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), curr_pent)

    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), np.flip(curr_pent))

    curr_pent = np.rot90(curr_pent)
    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), curr_pent)

    for i in range(shape[0]):
        for j in range(shape[1]):
            yield ((i, j), np.flip(curr_pent))


def generate_occupied_cells(pent, loc, rotate, flip, pent_dict=None, index=None, csp=None):
    squares = set()

    # Transform the pentomino
    own_pent = np.copy(pent)
    for i in range(rotate):
        own_pent = np.rot90(own_pent)
    for i in range(flip):
        own_pent = np.flip(own_pent)

    # Find the non-zero squares in the pentomino
    for i in range(own_pent.shape[0]):
        for j in range(own_pent.shape[1]):
            if own_pent[i, j] != 0:
                squares.add((i + loc[0], j + loc[1]))

    if len(squares & csp.valid_locations) == len(squares):
        if pent_dict is not None and index is not None:
            pent_dict[(index, rotate, flip)] = own_pent

        return frozenset(squares)
    else:
        return frozenset()

def ordered_domain_values(next_var, assignment, csp):
    possible_values = assignment.variable_assignment[next_var]  # This is a list of all possible variable assignments

    # LCA
    restore_dict = []  # list of (value, assignments_that_would_be_removed)

    for i, value in zip(range(len(possible_values)), possible_values):
        loc, rot, flip, occupied = value

        # squares_to_remove = set([tuple(l + np.array(loc)) for l in np.argwhere(csp.variables[next_var] != 0)])
        squares_to_remove = occupied

        restore_dict.append((value, defaultdict(set)))

        # From the other assignments
        for key in assignment.unassigned_vars:
            if key == next_var:
                continue

            # If they have potential assignments that would be made impossible, add them the the dict
            for unassigned_val in assignment.variable_assignment[key]:
                u_loc, u_rot, u_flip, u_occupied = unassigned_val
                u_pent = csp.variables[key]

                u_squares = u_occupied

                if len(squares_to_remove & u_squares) > 0:
                    # If they overlap
                    restore_dict[i][1][key].add(unassigned_val)

    sorted(restore_dict, key=lambda x: len(x[1]))

    for i in restore_dict:
        # pent = np.copy(csp.variables[next_var])
        value, removed_dict = i
        loc, rot, flip, occupied = value
        if len(occupied & csp.occupied_locations) == 0:
            pent = csp.rotated_pents[(next_var, rot, flip)]
            yield pent, loc, removed_dict, occupied


def consistent(value, assignment, problem):
    pent, loc, remove_dict, occupied = value

    # TODO: Need to prune here

    for i in remove_dict:
        if len(assignment.variable_assignment[i]) == len(remove_dict[i]):
            return False

    return True

def display_board(assignment, csp):
    display = np.copy(csp.board)
    for item in assignment.end_locations:
        if item is None:
            continue
        pent, loc = item
        for i in range(pent.shape[0]):
            for j in range(pent.shape[1]):
                if pent[i][j] != 0:
                    display[loc[0] + i, loc[1] + j] = pent[i][j]

    print(display)


import itertools
from collections import deque


def remove_inconsistent_values(var_one, var_two, assignment, csp):
    removed = False
    deleted = set()

    for x in assignment.variable_assignment[var_one]:
        x_loc, x_rot, x_flip, x_occ = x
        # x_occ = generate_occupied_cells(csp.variables[var_one], x_loc, x_rot, x_flip)

        valid_assignment = False
        for y in assignment.variable_assignment[var_two]:
            y_loc, y_rot, y_flip, y_occ = y
            # y_occ = generate_occupied_cells(csp.variables[var_two], y_loc, y_rot, y_flip)

            if len(x_occ & y_occ) == 0:
                valid_assignment = True
                break

        if not valid_assignment:
            removed = True
            deleted |= {x}

    assignment.variable_assignment[var_one] -= deleted
    return removed, deleted


def arc_consistency(assignment, csp):
    queue = deque(itertools.permutations(assignment.unassigned_vars, 2))

    removed_domains = defaultdict(set)

    while len(queue) > 0:
        arc = queue.popleft()

        removed, dele = remove_inconsistent_values(arc[0], arc[1], assignment, csp)
        removed_domains[arc[0]] |= dele

        if removed:
            for x_k in assignment.unassigned_vars:
                if x_k is not arc[0]:
                    queue.append((x_k, arc[0]))

    return removed_domains # form {var: domain}



def backtracking(assignment, csp):

    if goal_test(assignment) is True:
        return assignment

    next_var = select_unassigned_var(assignment, csp)

    for pent, loc, remove_dict, occupied in ordered_domain_values(next_var, assignment, csp):
        value = pent, loc, remove_dict, occupied

        removed = arc_consistency(assignment, csp)
        if consistent(value, assignment, csp):
            # TODO Fix below
            # assignment[next_var]=value
            assignment.is_assigned[next_var] = True
            assignment.unassigned_vars = assignment.unassigned_vars - {next_var}
            assignment.end_locations[next_var] = pent, loc

            # Forward propagation
            for i in remove_dict:
                assignment.variable_assignment[i] -= remove_dict[i]

            # pent_locs = []
            # for i in range(pent.shape[0]):
            #     for j in range(pent.shape[1]):
            #         if pent[i, j] != 0:
            #             pent_locs.append((i + loc[0], j + loc[1]))
            # pent_locs = frozenset(pent_locs)
            pent_locs = occupied
            csp.occupied_locations |= pent_locs # TODO: REIMPLEMENT OCCUPIED LOC

            result = backtracking(assignment, csp)
            # display_board(assignment, csp)
            if result is not None:
                return result

            assignment.is_assigned[next_var] = False
            assignment.unassigned_vars = assignment.unassigned_vars | {next_var}
            assignment.end_locations[next_var] = None

            for i in remove_dict:
                assignment.variable_assignment[i] |= remove_dict[i]
            csp.occupied_locations -= pent_locs
        for i in removed:
            assignment.variable_assignment[i] |= removed[i]

    return None # Failed


def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is 
    the coordinate of the upper left corner of pi in the board (lowest row and column index 
    that the tile covers).
    
    -Use np.flip and np.rot90 to manipulate pentominos.
    
    -You can assume there will always be a solution.
    """

    choices = {} # (pent, loc, rot, flip) :
    # Initial assignment, no pentiminos anywhere
    variable_assignments = {}

    rotated_dict = {}

    csp = ConstraintProblem(
        board.copy(),
        pents,
        frozenset([tuple(l) for l in np.argwhere(board == 1)]),
        set(),
        rotated_dict
    )
    for i in range(len(pents)):

        for row in range(board.shape[0] - pents[i].shape[0] + 1):
            for col in range(board.shape[1] - pents[i].shape[1] + 1):
                if i not in variable_assignments:
                    variable_assignments[i] = set()
                loc = row, col

                # Tuple of the form loc, num_rot, num_flip
                variable_assignments[i] |= {VarAssignment(loc, 0, 0, generate_occupied_cells(pents[i], loc, 0, 0, rotated_dict, i, csp))}
                variable_assignments[i] |= {VarAssignment(loc, 0, 1, generate_occupied_cells(pents[i], loc, 0, 1, rotated_dict, i, csp))}
                variable_assignments[i] |= {VarAssignment(loc, 2, 0, generate_occupied_cells(pents[i], loc, 2, 0, rotated_dict, i, csp))}
                variable_assignments[i] |= {VarAssignment(loc, 2, 1, generate_occupied_cells(pents[i], loc, 2, 1, rotated_dict, i, csp))}

        for row in range(board.shape[0] - pents[i].shape[1] + 1):
            for col in range(board.shape[1] - pents[i].shape[0] + 1):
                if i not in variable_assignments:
                    variable_assignments[i] = set()
                loc = row, col

                # Tuple of the form loc, num_rot, num_flip
                variable_assignments[i] |= {VarAssignment(loc, 1, 0, generate_occupied_cells(pents[i], loc, 1, 0, rotated_dict, i, csp))}
                variable_assignments[i] |= {VarAssignment(loc, 1, 1, generate_occupied_cells(pents[i], loc, 1, 1, rotated_dict, i, csp))}
                variable_assignments[i] |= {VarAssignment(loc, 3, 0, generate_occupied_cells(pents[i], loc, 3, 0, rotated_dict, i, csp))}
                variable_assignments[i] |= {VarAssignment(loc, 3, 1, generate_occupied_cells(pents[i], loc, 3, 1, rotated_dict, i, csp))}

    assignment = Assignment(np.array([False for _ in pents]),  # Current variable assignments
                            frozenset(range(len(pents))),  # Variables without assignments
                            [None for _ in pents],  # Variables locations
                            variable_assignments  # Possible assignments
                            )

    assignment = backtracking(assignment, csp)

    # Construct return
    # print(assignment.end_locations)
    # print(csp.valid_locations)
    # print(csp.occupied_locations)
    display_board(assignment, csp)
    return assignment.end_locations
