# -*- coding: utf-8 -*-
import numpy as np
import itertools
from collections import defaultdict, deque
import heapq

def transform_pent(pent, rotate, flip):
    own_pent = np.copy(pent)
    for i in range(rotate):
        own_pent = np.rot90(own_pent)
    for i in range(flip):
        own_pent = np.flip(own_pent)
    return own_pent


def generate_occupied_cells(pent, loc, rotate, flip, constraints, board_shape):
    squares = set()

    # Transform the pentomino
    own_pent = transform_pent(pent, rotate, flip)

    # Find the non-zero squares in the pentomino
    for i in range(own_pent.shape[0]):
        for j in range(own_pent.shape[1]):
            if own_pent[i, j] != 0:
                squares.add((i + loc[0], j + loc[1]))

    if loc[0] + own_pent.shape[0] <= board_shape[0] and loc[1] + own_pent.shape[1] <= board_shape[1] and len(squares & constraints) == 0:
        # print('HI!')
        return frozenset(squares)
    else:
        return frozenset()


def select_unassigned_var(variables, unused_pents, pent_constraints,
                          location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict):
    min_val = len(variables[min(unassigned_vars, key=lambda x: len(variables[x]))])
    min_list = list(filter(lambda x: len(variables[x]) == min_val, variables))

    if len(min_list) == 1:
        return min_list[0]

    # MCV
    # for poss in min_list:
    #     # Count the number of times min occurs in the others vars
    # for unassigned in unassigned_vars:
    #     if abs(arc[0][0] - arc[1][0]) > 5 or abs(arc[0][1] - arc[1][1]) > 5:
    #         continue

    return min_list[0]


def shares_pent(assign_one, assign_two):
    return assign_one[0] == assign_two[0]


def is_same_assignment(assign_one, assign_two):
    return assign_one == assign_two


def overlap(assign_one, assign_two, pents_occupied, overlap_dict=None):
    if overlap_dict is None:
        return is_same_assignment(assign_one, assign_two) or \
               any(x in pents_occupied[assign_one] for x in pents_occupied[assign_two])
               # len(pents_occupied[assign_one] & pents_occupied[assign_two]) > 0

    if (assign_one, assign_two) in overlap_dict:
        return overlap_dict[(assign_one, assign_two)]

    return overlap_dict[(assign_two, assign_one)]


def ordered_domain_values(next_var, variables, unused_pents, pent_constraints, location_constraints,
                          assignment, pents_occupied, unassigned_vars, overlap_dict):
    print('Ordered')
    # LCA - Least Constraining Assignment
    # next_var is the location on the square to fill
    domain_values = variables[next_var]

    restore_dict_dict = {}  # dict[domain_value] = dict[variable] = {removed domain values})
    restore_count = {}  # dict[domain_value] = count
    for i, move_id in zip(range(len(domain_values)), domain_values):
        # Find least constraining domain_value
        # move_id = (i, loc, rot, flip)
        constraints = pents_occupied[move_id]

        # For now simple count of the number of constraints removed from the neighboring squares
        restore_dict_dict[move_id] = defaultdict(set)
        restore_count[move_id] = 0

        for unassigned_var in unassigned_vars:
            if unassigned_var is next_var:
                # If it is the variable it eliminates all possible domains
                removed = variables[next_var]
                restore_dict_dict[move_id][unassigned_var] |= removed
                restore_count[move_id] += len(removed)
                continue

            if unassigned_var in location_constraints:
                # Ignore invalid locations
                continue

            # Let us now see from the unassigned vars, how many constraints we impose on them
            unassigned_domain = variables[unassigned_var]

            # Do the pents created from the assignments overlap, do they use the same pent is the
            # Do the domain values overlap or use the same pentomino
            # x is just occupied locations
            removed = set(filter(lambda x: shares_pent(x, move_id)
                                or overlap(x, move_id, pents_occupied, overlap_dict), unassigned_domain))

            restore_dict_dict[move_id][unassigned_var] |= removed
            restore_count[move_id] += len(removed)

    heap = [(-value, key) for key, value in restore_count.items()]
    # sorted_count = sorted(restore_count, key=restore_count.get)
    # sorted_count.reverse()
    # for move_id in sorted_count:
    #     # pent = np.copy(csp.variables[next_var])
    #     yield move_id, restore_dict_dict[move_id]
    heapq.heapify(heap)
    print('Ordered end')
    while len(heap) > 0:
        _, move_id = heapq.heappop(heap)
        yield move_id, restore_dict_dict[move_id]


def remove_inconsistent_values(var_one, var_two, variables, unused_pents, pent_constraints,
                    location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict):
    # print('')
    removed = False
    deleted = set()

    for x in variables[var_one]: # Get the domain of var_one (its possible assignments)
        # if no value in var_one domain allows (var_one, var_two) to allows these two
        # to coexist
        # x = (i, loc, 0, 0)
        valid_assignment = False

        for y in variables[var_two]:
            if is_same_assignment(x, y):
                valid_assignment = True
                break

            if overlap(x, y, pents_occupied, overlap_dict) == 0:
                valid_assignment = True
                break

        if not valid_assignment:
            removed = True
            deleted |= {x}

    variables[var_one] -= deleted
    return removed, deleted


def arc_consistency(variables, unused_pents, pent_constraints,
                    location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict):
    print('Arc Consistency')
    queue = deque(itertools.permutations(unassigned_vars, 2))

    removed_domains = defaultdict(set)

    while len(queue) > 0:
        arc = queue.popleft()

        # TODO: FIX
        if abs(arc[0][0] - arc[1][0]) > 5 or abs(arc[0][1] - arc[1][1]) > 5:
                continue

        removed, dele = remove_inconsistent_values(arc[0], arc[1], variables, unused_pents, pent_constraints,
                    location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict)

        # Archive the removed pent assignments
        removed_domains[arc[0]] |= dele

        # Need to re-evalute arcs
        if removed:
            for x_k in unassigned_vars:
                if x_k is not arc[0]:
                    queue.append((x_k, arc[0]))
    print('Arc Consistency END')
    return removed_domains  # form {var: domain}


def consistent(variables, unused_pents, pent_constraints,
                    location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict):

    for var in unassigned_vars:
        if len(variables[var]) == 0:
            return False

    return True


def display_board(assignment, pents, board):
    display = np.copy(board)
    # for pent, loc in assignment:
    for item in assignment:
        if item is None:
            continue
        pent, loc = item
        for i in range(pent.shape[0]):
            for j in range(pent.shape[1]):
                if pent[i][j] != 0:
                    display[loc[0] + i, loc[1] + j] = pent[i][j]
    print(display)


def backtracking(variables, unused_pents, pent_constraints,
                 location_constraints, assignment, pents_occupied, unassigned_vars,
                 pents, board, overlap_dict):
    """
    Consistency will remove from the domain
    :param variables:
    :param pents_constraints:
    :param location_constraints:
    :param assignment:
    :return:
    """
    if len(unassigned_vars) == 0:
        return assignment

    next_var = select_unassigned_var(variables, unused_pents, pent_constraints,
                                     location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict)

    for move_id, first_order_removal in ordered_domain_values(next_var, variables, unused_pents,
                                                              pent_constraints, location_constraints,
                                                              assignment, pents_occupied, unassigned_vars,
                                                              overlap_dict):
        # move_id = (i, loc, rot, flip)
        # ordered_domain_values(next_var, assignment, csp):
        # First step removal
        # Remove from simply picking value
        #   - Mark locations as occupied
        #   - Mark the pent as used
        unused_pents -= {move_id[0]}
        pent_constraints |= {move_id[0]}
        location_constraints |= {next_var}
        unassigned_vars -= pents_occupied[move_id]

        # print('1')
        for i in first_order_removal:
            variables[i] -= first_order_removal[i]

        assert (len(variables[next_var]) == 0)

        # Second Order - Arc consistency
        if consistent(variables, unused_pents, pent_constraints,
                      location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict):
            arc_consistency_removal = arc_consistency(variables, unused_pents, pent_constraints,
                                         location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict)
            # for i in arc_consistency_removal:
            #     assignment.variable_assignment[i] -= arc_consistency_removal[i]
            # print('2')
            if consistent(variables, unused_pents, pent_constraints,
                        location_constraints, assignment, pents_occupied, unassigned_vars, overlap_dict):
                # print('3')
                # TODO Fix below
                # [(p1, (row1, col1))...(pn, (rown, coln))]

                own_pent = transform_pent(pents[move_id[0]], move_id[2], move_id[3])
                assignment.append((own_pent, move_id[1]))  # pent, loc
                # assignment.append((pents[move_id[0]], move_id[1]))  # pent, loc
                display_board(assignment, pents, board)
                result = backtracking(variables, unused_pents, pent_constraints,
                        location_constraints, assignment, pents_occupied, unassigned_vars, pents, board, overlap_dict)

                if result is not None:
                    return result

                del assignment[-1]

            # Clean up after arc consistency
            for i in arc_consistency_removal:
                variables[i] |= arc_consistency_removal[i]
        

        # Clean up first order removal
        for i in first_order_removal:
            variables[i] |= first_order_removal[i]

        unused_pents |= {move_id[0]}
        pent_constraints -= {move_id[0]}
        location_constraints -= {next_var}
        unassigned_vars |= pents_occupied[move_id]

    return None  # Failed


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

    variables = {tuple(i): set() for i in np.argwhere(board == 1)}  # variables [i] is the domain
    # unassigned_variables = {tuple(i): set() for i in np.argwhere(board == 1)}
    unused_pents = {i for i in range(len(pents))}
    pents_constraints = set()  # This is where all used pents go
    location_constraints = {tuple(i) for i in np.argwhere(board == 0)}  # set of invalid locations

    pents_occupied = {}

    for loc in itertools.product(range(board.shape[0]), range(board.shape[1])):
        for i in range(len(pents)):
            pents_occupied[(i, loc, 0, 0)] = generate_occupied_cells(pents[i], loc, 0, 0, location_constraints, board.shape)
            pents_occupied[(i, loc, 1, 0)] = generate_occupied_cells(pents[i], loc, 1, 0, location_constraints, board.shape)
            pents_occupied[(i, loc, 2, 0)] = generate_occupied_cells(pents[i], loc, 2, 0, location_constraints, board.shape)
            pents_occupied[(i, loc, 3, 0)] = generate_occupied_cells(pents[i], loc, 3, 0, location_constraints, board.shape)
            pents_occupied[(i, loc, 0, 1)] = generate_occupied_cells(pents[i], loc, 0, 1, location_constraints, board.shape)
            pents_occupied[(i, loc, 1, 1)] = generate_occupied_cells(pents[i], loc, 1, 1, location_constraints, board.shape)
            pents_occupied[(i, loc, 2, 1)] = generate_occupied_cells(pents[i], loc, 2, 1, location_constraints, board.shape)
            pents_occupied[(i, loc, 3, 1)] = generate_occupied_cells(pents[i], loc, 3, 1, location_constraints, board.shape)

    # overlap_dict = {}
    # for combo in itertools.combinations_with_replacement(pents_occupied.keys(), 2):
    #     overlap_dict[combo] = overlap(combo[0], combo[1], pents_occupied)
    overlap_dict=None
    print('Hi!')

    for loc in itertools.product(range(board.shape[0]), range(board.shape[1])):
        for combo in pents_occupied:  # The combo
            occ_set = pents_occupied[combo]
            if loc in occ_set:
                variables[loc].add(combo)
    assignment = []
    backtracking(variables, unused_pents, pents_constraints, location_constraints,
                 assignment, pents_occupied, variables.keys(), pents, board, overlap_dict)


    return assignment
