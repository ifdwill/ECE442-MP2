import numpy as np
import itertools
from collections import deque, defaultdict


# Indentation style is tabs

class Assignment:
    """
    This class represents a possible choice (placements) that the variables (pentominos) have
    """

    def __init__(self, pent_idx, location, rotation, flip, csp):
        """This is a more constrained definition of assignment
        pent - numpy array representing the pentomino
        location - tuple (row, col)
        rotation - How many times the pentomino has been rotated [0, 3] (applied before flipping)
        flip - How mant times the pentomino has been flipped. [0, 1]
        csp  - A handle to access the csp to determine if any locations are invalid
        """

        # Rotate the pent (Constant after this DO NOT MODIFY)
        self.pent_idx = pent_idx
        self.pent = np.copy(csp.variables[pent_idx])
        # pent = self.pent

        for _ in range(rotation):
            self.pent = np.rot90(self.pent)
        for _ in range(flip):
            self.pent = np.flip(self.pent)

        self.shape = self.pent.shape

        # Set the location
        self.location = location
        self.rotation = rotation
        self.flip = flip

        # Game awareness
        self.csp = csp

        # This consists of (Just useful information to keep around)
        self.occupied = self.generate_occupied_cells(self.pent, self.location)

        # The cosntraints that this assignment imposes
        self.constraints_set = set()

    def generate_occupied_cells(self, pent, loc):
        squares = set()
        loc = self.location

        # print(self.shape)
        # print(pent)
        # Find the non-zero squares in the pentomino
        for i, j in itertools.product(range(self.shape[0]), range(self.shape[1])):
            if self.pent[i, j] != 0:
                squares.add((i + loc[0], j + loc[1]))
        # for i in range(self.shape[0]):
        #     for j in range(self.shape[1]):
        #         if self.pent[i, j] != 0:
        #             squares.add((i + loc[0], j + loc[1]))

        if all(self.csp.in_bound(sq) for sq in squares):
            return frozenset(squares)
        else:
            return frozenset()

    def __eq__(self, other):
        if isinstance(other, Assignment):
            return self.pent_idx == other.pent_idx and \
                self.location == other.location and \
                self.rotation == other.rotation and \
                self.flip == other.flip

        return False

    def __hash__(self):
        # print('The hash is:')
        return hash((self.pent_idx, self.location, self.rotation, self.flip))


class Constraint:

    def __init__(self, constraint=None, is_pent=False):
        # pent is the pentomino used in the constraint
        # Can either be an int (pent constraint) or a (int, int) (a location constraint)
        self.constraint = constraint

        # This is the list of assignments that the constaint ties down
        self.assignments_set = set()


class CSP:
    """
    This class represents the problem domain containing abstract information about the game itself.
    As well, it contains the two way dictionary between constraints and assignments and mapping from a variable
    to its domain.
    """

    def __init__(self, board, pents):
        self.board = board
        self.shape = board.shape
        self.variables = pents
        # An index for each pentomino (This is by)

        self.unassigned_vars = set(i for i in range(len(pents)))
        self.variables_indices = frozenset(i for i in range(len(pents)))
        # This contains a mapping from variables indices to their domain (set of assignments)
        self.domains = {}
        self.constraints = {}

        self.init_domain()
        self.init_constr_assign()

        self.generate_connections()

    def init_domain(self):
        for pent_idx in self.variables_indices:
            self.domains[pent_idx] = set()

        for pent_idx in self.variables_indices:
            # Generate all possible assignments
            for temp_assign in itertools.product(range(self.shape[0]), range(self.shape[1]), range(4), range(2)):
                loc = (temp_assign[0], temp_assign[1])
                rot = temp_assign[2]
                flip = temp_assign[3]

                assign = Assignment(pent_idx, loc, rot, flip, self)

                if len(assign.occupied) != 0:
                    # Valid assignment
                    self.domains[pent_idx].add(assign)

    def init_constr_assign(self):
        # First create indices
        for idx in self.variables_indices:
            self.constraints[idx] = Constraint(idx)

        for loc in itertools.product(range(self.shape[0]), range(self.shape[1])):
            self.constraints[loc] = Constraint(loc)

    def generate_connections(self):
        for pent_idx in self.domains:
            for assign in self.domains[pent_idx]:

                # Pent Constriant
                pent_constraint = self.constraints[pent_idx]
                pent_constraint.assignments_set.add(assign)
                assign.constraints_set.add(pent_constraint)

                # Location Constraint
                for location in assign.occupied:
                    loc_constraint = self.constraints[location]
                    loc_constraint.assignments_set.add(assign)
                    assign.constraints_set.add(loc_constraint)

    def in_bound(self, location):
        return location[0] < self.shape[0] and location[1] < self.shape[1] \
            and location[0] >= 0 and location[1] >= 0

    def assign_var(self, var_idx):
        pass

    # def violate_constrain(assignment, constraint):


def display_board(csp, end_locations):
    display = np.copy(csp.board)

    assignment = []
    for i in end_locations:
        # print(i)
        assignment.append((i.pent, i.location))

    for assign in end_locations:

        for i in range(assign.pent.shape[0]):
            for j in range(assign.pent.shape[1]):
                if assign.pent[i][j] != 0:
                    display[assign.location[0] + i,
                            assign.location[1] + j] = -assign.pent[i][j]

    print(display)


def goal_test(csp):
    return len(csp.unassigned_vars) == 0


def select_unassigned_var(csp):
    """Uses least remaining values as its initial filter. If there is a tie
    then the algorithm breaks it with most constraining variable. The remaining tie
    is simply broken arbitrarily."""

    min_val = len(
        csp.domains[min(csp.unassigned_vars, key=lambda x: len(csp.domains[x]))])
    # print(min_val)
    min_list = list(filter(lambda x: len(
        csp.domains[x]) == min_val, csp.unassigned_vars))

    if len(min_list) == 1:
        return min_list[0]

    # MCV
    # min_list is a list of variables
    # check the number of neighboring pents

    def count_constraint(x):
        # print(csp.unassigned_vars)
        # print(min_list)
        # print(x)
        # if type(x) is set:
        #     for i in x:
        #         print(i)
        assert(x in min_list)
        assert(x in csp.unassigned_vars)
        assert(type(i) is not set for i in min_list)
        # print (x)
        # assert(type(x) is not set)
        # print(csp.domains[x])
        # assert(type(i) is not set for i in csp.domains[x])
        # print(list(assign.constraints_set for assign in csp.domains[x]))
        return sum(len(assign.constraints_set) for assign in csp.domains[x])

    # FIXME:
    most_constraints = count_constraint(max(min_list, key=count_constraint))

    max_list = list(filter(lambda x: count_constraint(x)
                           == most_constraints, min_list))

    return max_list[0]


def ordered_domain_values(next_var, csp):
    """ This function orders the values within the assignments from next_var's domain
    next_var - the next assigned variable (index)
    csp - the constraint problem for easy access"""

    # LCA - Least Constraining Assignment
    # next_var is the location on the square to fill
    domain_values = list(csp.domains[next_var])

    sorted(domain_values, key=lambda x: len(x.constraints_set))

    for assignment in domain_values:
        # FIXME: DIRE

        # Dictionary from variable -> assignments removed
        assignment_removal_dict = {}

        a = sum([len(csp.domains[i]) for i in csp.unassigned_vars]) 
        
        # Remove assignments from domain.
        for c in assignment.constraints_set:
            # Propogate constraints
            for neighbor_assignment in c.assignments_set:
                neighbor = neighbor_assignment.pent_idx
                if neighbor not in assignment_removal_dict:
                    assignment_removal_dict[neighbor] = set()

                assignment_removal_dict[neighbor].add(neighbor_assignment)
        
        for neighbor in assignment_removal_dict:
            csp.domains[neighbor] -= assignment_removal_dict[neighbor]

        yield assignment, assignment_removal_dict

        # Restore constraint and locations
        for neighbor in assignment_removal_dict:
            # Propogate constraints
            print(type(csp.domains[neighbor]))
            print(type(assignment_removal_dict[neighbor]))
            csp.domains[neighbor] |= assignment_removal_dict[neighbor]

        b = sum([len(csp.domains[i]) for i in csp.unassigned_vars]) 
        assert(a == b)

    pass


def consistent(csp):
    for var in csp.unassigned_vars:
        if len(csp.domains[var]) == 0:
            return False

    return True


def remove_inconsistent_values(var_one, var_two, csp):
    removed = False
    deleted = set()

    # Get the domain of var_one (its possible assignments)
    for x in csp.domains[var_one]:
        # if no value in var_one domain allows (var_one, var_two) to allows these two
        # to coexist
        # x = (i, loc, 0, 0)
        valid_assignment = False

        for y in csp.domains[var_two]:
            if x == y:
                valid_assignment = True
                break

            if not any(i in y.constraints_set for i in x.constraints_set):
                valid_assignment = True
                break

        if not valid_assignment:
            removed = True
            deleted |= {x}

    csp.domains[var_one] -= deleted
    return removed, deleted


def arc_consistency(csp):
    print('Arc Consistency')
    seen_edges = set()
    for i in csp.unassigned_vars:
        # Now iterate through domain of unassigned var
        domain = csp.domains[i]
        for assignment in domain:
            for j in assignment.constraints_set:
                for k in j.assignments_set:
                    if i != k.pent_idx:
                        seen_edges.add((i, k.pent_idx))
                        # print(k)
                        assert(type(i) is int)
                        assert(type(k.pent_idx) is int)
    queue = deque(seen_edges)

    removed_domains = defaultdict(set)

    while len(queue) > 0:
        arc = queue.popleft()

        removed, dele = remove_inconsistent_values(arc[0], arc[1], csp)

        # Archive the removed pent assignments
        removed_domains[arc[0]] |= dele

        # Need to re-evalute arcs
        if removed:
            for x_k in csp.unassigned_vars:
                if x_k is not arc[0] and (x_k, arc[0]) in seen_edges:
                    queue.append((x_k, arc[0]))
    print('Arc Consistency END')
    return removed_domains  # form {var: domain}


import copy

def backtracking(csp, end_locations):

    if goal_test(csp) is True:
        return end_locations

    # print("a")
    next_var = select_unassigned_var(csp)
    # print("b")
    a = sum([len(csp.domains[i]) for i in csp.unassigned_vars]) 
    for assignment, removed_dict in ordered_domain_values(next_var, csp):
        # print("c")
        csp.unassigned_vars.remove(next_var)
        if consistent(csp):
            # TODO Fix below
            # print("d")
            end_locations.append(assignment)

            print(next_var)
            display_board(csp, end_locations)
            result = backtracking(csp, end_locations)
            
            if result is not None:
                return result

            end_locations.pop()


        b = sum([len(csp.domains[i]) for i in csp.unassigned_vars])
        print(a, b)
        # assert(a == b)


    b = sum([len(csp.domains[i]) for i in csp.unassigned_vars]) 
    assert(a == b)
    print('Failure')
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

    csp = CSP(board, pents)

    assignment = backtracking(csp, [])

    # Construct return
    # print(assignment.end_locations)
    # print(csp.valid_locations)
    # print(csp.occupied_locations)
    arr = []
    for i in assignment:
        arr.append((i.pent, i.location))

    # display_board(csp)
    return arr
