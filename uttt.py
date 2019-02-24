from time import sleep
#from math import inf
from random import randint
import itertools
import copy
import sys



class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        #self.startBoardIdx=4
        self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True

        # Custom member
        self.bestMoveFound = None
        self.currBoardIdx = self.globalIdx[self.startBoardIdx]

	#def assignScoreDesigned(self, l, isMax, checkWinner): 

    def assign_score(self, l, isMax, maxPlayer = 'X', minPlayer = 'O'):
        score = 0
            
        if isMax :
            if l.count(maxPlayer) == 2 : # 2 X's in the row/col/diag
                if l.count(minPlayer) == 0 : #unblocked X 
                    score += 500
            if l.count(minPlayer) == 2:
                if l.count(maxPlayer) == 1:
                    score +=100
        else : #minplayer
            if l.count(minPlayer) == 2 and l.count(maxPlayer) == 0:
                score -= 100
            if l.count(maxPlayer) == 2 and l.count(minPlayer) == 1:
                score -=500
        return score
			
    def boardValue(self, currentBoard, isMax, emptySpace = '_', maxPlayer = 'X', minPlayer = 'O'):
        score = 0
        if isMax:
            if currentBoard < 3 :
               for i in range(3) :
                   for j in range(3):
                       if self.board[3*i][(3*j)+ currentBoard] == emptySpace :
                           score += 100
            elif currentBoard >=3 and currentBoard < 6 :
                for i in range(3) :
                    for j in range(3):
                        if self.board[3*i + 1][(3*j) + (currentBoard%3)] == emptySpace :
                            score += 100
            elif currentBoard >=6 and currentBoard < 9 :
                for i in range(3) :
                    for j in range(3):
                        if self.board[3*i + 2][(3*j) + (currentBoard%3)] == emptySpace :
                            score += 100
        else:
            if currentBoard < 3 :
               for i in range(3) :
                   for j in range(3):
                       if self.board[3*i][(3*j)+ currentBoard] == emptySpace :
                           score -= 100
            elif currentBoard >=3 and currentBoard < 6 :
                for i in range(3) :
                    for j in range(3):
                        if self.board[3*i + 1][(3*j) + (currentBoard%3)] == emptySpace :
                            score -= 100
            elif currentBoard >=6 and currentBoard < 9 :
                for i in range(3) :
                    for j in range(3):
                        if self.board[3*i + 2][(3*j) + (currentBoard%3)] == emptySpace :
                            score -= 100
        return score    

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        #print(isMax)
        winner = self.checkWinner()
        #"""
        if winner == 1  and isMax:
            return 10000
        if winner == -1 and not isMax:
            return -10000
        #"""
        score = 0
        
        for local_board_x, local_board_y in self.globalIdx:
            # Iterate through local boards
            for i in range(3):
                # check for a row with the same vals
                row = self.board[local_board_x + i][local_board_y: local_board_y + 3]
                score += self.assign_score(row,isMax)

            # Check cols
            for i in range(3):
                # check for a row with the same vals
                col = [self.board[local_board_x + j][local_board_y + i] for j in range(3)]
                score += self.assign_score(col, isMax)

            diag = [self.board[local_board_x + i][local_board_y + i] for i in range(3)]
            score += self.assign_score(diag, isMax)
            
            diag = [self.board[local_board_x + i][local_board_y + 2 - i] for i in range(3)]
            score += self.assign_score(diag, isMax)

        if score != 0:
            #print('Rule 2 Score:', score)
            return score

        # Rule 3:
        for local_board_x, local_board_y in self.globalIdx:
            corners = [self.board[local_board_x][local_board_y], self.board[local_board_x + 2][local_board_y],
                       self.board[local_board_x][local_board_y + 2], self.board[local_board_x + 2][local_board_y + 2]]
            if isMax:
                score += (30 * corners.count(self.maxPlayer))
            else:
                score -= (30 * corners.count(self.minPlayer))
        #print('Rule 3 score:', score)
        return score

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0

        winner = self.checkWinner()
        if winner == 1  and isMax:
            return 100000
        if winner == -1 and not isMax:
            return -100000
        
        score = 0
        #curr_board = self.globalIdx[self.currBoardIdx]
        #print('Evaluate Designed currBoardIdx' , self.currBoardIdx)
        #print('curr_board: ' , curr_board)
        for i in range(9):
            #score += self.boardValue(self.currBoardIdx, isMax)
            score += self.boardValue(i, isMax)
        
        for local_board_x, local_board_y in self.globalIdx:
            # Iterate through local boards
            
            for i in range(3):
                # check for a row with the same vals
                row = self.board[local_board_x + i][local_board_y: local_board_y + 3]
                score += self.assign_score(row,isMax)

            # Check cols
            for i in range(3):
                # check for a row with the same vals
                col = [self.board[local_board_x + j][local_board_y + i] for j in range(3)]
                score += self.assign_score(col, isMax)

            diag = [self.board[local_board_x + i][local_board_y + i] for i in range(3)]
            score += self.assign_score(diag, isMax)
            
            diag = [self.board[local_board_x + i][local_board_y + 2 - i] for i in range(3)]
            score += self.assign_score(diag, isMax)

        if score != 0:
            #print('Rule 2 Score:', score)
            return score

        # Rule 3:
        for local_board_x, local_board_y in self.globalIdx:
            corners = [self.board[local_board_x][local_board_y], self.board[local_board_x + 2][local_board_y],
                       self.board[local_board_x][local_board_y + 2], self.board[local_board_x + 2][local_board_y + 2]]
            if isMax:
                score += (30 * corners.count(self.maxPlayer))
            else:
                score -= (30 * corners.count(self.minPlayer))
        #print('Rule 3 score:', score)
        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        movesLeft=False
        for i in range(len(self.board)):
            movesLeft |= '_' in self.board[i]

        return movesLeft

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        #YOUR CODE HERE
        # winner = 0

        for local_board_x, local_board_y in self.globalIdx:
            # Iterate through local boards

            for i in range(3):
                # check for a row with the same vals
                if self.board[local_board_x + i][local_board_y] \
                        == self.board[local_board_x + i][local_board_y + 1] \
                        == self.board[local_board_x + i][local_board_y + 2]:

                    winner = 1 if self.board[local_board_x + i][local_board_y] == self.maxPlayer else 0
                    winner = -1 if self.board[local_board_x + i][local_board_y] == self.minPlayer else winner

                    if winner != 0:
                        return winner

            # Check cols
            for i in range(3):
                # check for a row with the same vals
                if self.board[local_board_x + 0][local_board_y + i] \
                        == self.board[local_board_x + 1][local_board_y + i] \
                        == self.board[local_board_x + 2][local_board_y + i]:

                    winner = 1 if self.board[local_board_x][local_board_y + i] == self.maxPlayer else 0
                    winner = -1 if self.board[local_board_x][local_board_y + i] == self.minPlayer else winner

                    if winner != 0:
                        return winner

            if self.board[local_board_x + 0][local_board_y + 0] \
                    == self.board[local_board_x + 1][local_board_y + 1] \
                    == self.board[local_board_x + 2][local_board_y + 2]:
                winner = 1 if self.board[local_board_x + 1][local_board_y + 1] == self.maxPlayer else 0
                winner = -1 if self.board[local_board_x + 1][local_board_y + 1] == self.minPlayer else winner

                if winner != 0:
                    return winner

            if self.board[local_board_x + 0][local_board_y + 2] \
                    == self.board[local_board_x + 1][local_board_y + 1] \
                    == self.board[local_board_x + 2][local_board_y + 0]:
                winner = 1 if self.board[local_board_x + 1][local_board_y + 1] == self.maxPlayer else 0
                winner = -1 if self.board[local_board_x + 1][local_board_y + 1] == self.minPlayer else winner

                if winner != 0:
                    return winner

        return 0

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        # assert(False)
        isX = (isMax and depth%2 == 0) or (not isMax and depth%2 == 1)
        #print(isX)
        if depth == 3 or self.checkWinner() != 0:
            score = self.evaluatePredifined(isMax if depth%2 == 0 else not isMax)
            #print("depth 3 score", score)
            #self.printGameBoard()
            #if (isMax if depth%2 == 0 else not isMax) is False:
              #  return -1 * score
            return score 

        self.expandedNodes += 1
        curr_board = self.globalIdx[currBoardIdx]

        local_board = [(i, j) for i, j in itertools.product(range(3), range(3))]
        
        #value = -1E10 if depth%2 == 0 else 1E10
        value = None
        if isX:
            # value = 0
            for loc, idx in zip(local_board, range(len(local_board))):
                if self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] != '_':
                    # Illegal moves are not worth anything
                    continue

                # General case: Add character, call sub function, tear down
                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = \
                    self.maxPlayer if isMax else self.minPlayer
                if value == None :
                    value = self.alphabeta(depth+1, idx, alpha, beta, not isMax)
                else:
                    if isMax:
                        value = max(value, self.alphabeta(depth + 1, idx, alpha, beta, not isMax))
                    else:
                        value = min(value, self.alphabeta(depth + 1, idx, alpha, beta, not isMax))

                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = '_'

                if depth%2 == 0:
                # Alpha represents the minimum assured score for the maximizing player
                # if depth == 0:
                    # print('alpha:', alpha)
                    # print('value:', value)
                    if depth == 0 and (value > alpha or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    alpha = max(alpha, value)

                else:
                # Beta represents the maximum assured score for the minimizing player
                    if depth == 0 and (value < beta or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    beta = min(beta, value)

                if alpha >= beta:
                    break
        else :
            # value = 0
            for loc, idx in zip(local_board, range(len(local_board))):
                if self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] != '_':
                    # Illegal moves are not worth anything
                    continue

                # General case: Add character, call sub function, tear down
                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = \
                    self.maxPlayer if isMax else self.minPlayer
                if value == None :
                    value = self.alphabeta(depth+1, idx, alpha, beta, not isMax)
                else:
                    if isMax:
                        #print("if isMax value: ", value)
                        value = max(value, self.alphabeta(depth + 1, idx, alpha, beta, not isMax))
                    else:
                        #print("if isMax value: ", value)
                        value = min(value, self.alphabeta(depth + 1, idx, alpha, beta, not isMax))

                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = '_'

                if depth%2 == 0:
                # Alpha represents the minimum assured score for the maximizing player
                # if depth == 0:
                    # print('alpha:', alpha)
                    # print('value:', value)
                    if depth == 0 and (value < -alpha or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    alpha = -min(-alpha, value)

                else:
                # Beta represents the maximum assured score for the minimizing player
                    if depth == 0 and (value > -beta or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    beta = -max(-beta, value)

                if alpha >= beta:
                    break
        # if depth == 0:
        #     self.currBoardIdx = idx
        #if depth == 2 or depth == 1:
            #print("depth %d score %d"%(depth, value))
        return value

    def alphabeta_designed(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        # assert(False)
        isX = (isMax and depth%2 == 0) or (not isMax and depth%2 == 1)
        #print(isX)
        if depth == 3 or self.checkWinner() != 0:
            score = self.evaluateDesigned(isMax if depth%2 == 0 else not isMax)
            #print("depth 3 score", score)
            #self.printGameBoard()
            #if (isMax if depth%2 == 0 else not isMax) is False:
              #  return -1 * score
            return score 

        self.expandedNodes += 1
        curr_board = self.globalIdx[currBoardIdx]

        local_board = [(i, j) for i, j in itertools.product(range(3), range(3))]
        
        #value = -1E10 if depth%2 == 0 else 1E10
        value = None
        if isX:
            # value = 0
            for loc, idx in zip(local_board, range(len(local_board))):
                if self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] != '_':
                    # Illegal moves are not worth anything
                    continue

                # General case: Add character, call sub function, tear down
                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = \
                    self.maxPlayer if isMax else self.minPlayer
                if value == None :
                    value = self.alphabeta_designed(depth+1, idx, alpha, beta, not isMax)
                else:
                    if isMax:
                        value = max(value, self.alphabeta_designed(depth + 1, idx, alpha, beta, not isMax))
                    else:
                        value = min(value, self.alphabeta_designed(depth + 1, idx, alpha, beta, not isMax))

                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = '_'

                if depth%2 == 0:
                # Alpha represents the minimum assured score for the maximizing player
                # if depth == 0:
                    # print('alpha:', alpha)
                    # print('value:', value)
                    if depth == 0 and (value > alpha or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    alpha = max(alpha, value)

                else:
                # Beta represents the maximum assured score for the minimizing player
                    if depth == 0 and (value < beta or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    beta = min(beta, value)

                if alpha >= beta:
                    break
        else :
            # value = 0
            for loc, idx in zip(local_board, range(len(local_board))):
                if self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] != '_':
                    # Illegal moves are not worth anything
                    continue

                # General case: Add character, call sub function, tear down
                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = \
                    self.maxPlayer if isMax else self.minPlayer
                if value == None :
                    value = self.alphabeta_designed(depth+1, idx, alpha, beta, not isMax)
                else:
                    if isMax:
                        #print("if isMax value: ", value)
                        value = max(value, self.alphabeta_designed(depth + 1, idx, alpha, beta, not isMax))
                    else:
                        #print("if isMax value: ", value)
                        value = min(value, self.alphabeta_designed(depth + 1, idx, alpha, beta, not isMax))

                self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = '_'

                if depth%2 == 0:
                # Alpha represents the minimum assured score for the maximizing player
                # if depth == 0:
                    # print('alpha:', alpha)
                    # print('value:', value)
                    if depth == 0 and (value < -alpha or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    alpha = -min(-alpha, value)

                else:
                # Beta represents the maximum assured score for the minimizing player
                    if depth == 0 and (value > -beta or self.bestMoveFound is None):
                    # print('HI!')
                        self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                        self.currBoardIdx = idx
                    beta = -max(-beta, value)

                if alpha >= beta:
                    break
        # if depth == 0:
        #     self.currBoardIdx = idx
        #if depth == 2 or depth == 1:
            #print("depth %d score %d"%(depth, value))
        return value

    def minimax(self, depth, currBoardIdx, isMax):
        """
		#60, -60, 90, -90, 500, -100, 500, -120, 10000
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        if depth == 3 or self.checkWinner() != 0:
            # assume depth starts at 0
            # Note: This is not a meaningful state, but rather a simple way to do some function evals
            #print(isMax if depth%2==0 else not isMax)
            return self.evaluatePredifined(isMax if depth%2 == 0 else not isMax)

        self.expandedNodes += 1
        curr_board = self.globalIdx[currBoardIdx]
        local_board = [(i, j) for i, j in itertools.product(range(3), range(3))]
        move_vals = []
		
        for loc, idx in zip(local_board, range(len(local_board))):
            # For every local square
            if self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] != '_':
                # Illegal moves are not worth anything
                move_vals.append(None)
                continue

            # General case: Add character, call sub function, tear down
            self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = self.maxPlayer if isMax else self.minPlayer
            # print(idx)
            value = self.minimax(depth + 1, idx, not isMax)
            self.board[curr_board[0] + loc[0]][curr_board[1] + loc[1]] = '_'
            move_vals.append(value)

        #print(move_vals)
        bestValue = 0.0
        #print(depth)
        if isMax:
			if depth == 0 :
			    print("isMax Move Vals: " , move_vals)
			bestValue = max(x for x in move_vals if x is not None)
        else:
			#print("not isMax Move Vals: " , move_vals)
			bestValue = min(x for x in move_vals if x is not None)

        if depth == 0:
            for idx in range(len(move_vals)):
                if move_vals[idx] is not None and move_vals[idx] == bestValue:
                    loc = local_board[idx]
                    self.bestMoveFound = (curr_board[0] + loc[0], curr_board[1] + loc[1])
                    self.currBoardIdx = idx
                    break

        return bestValue

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        bestValue=[]
        expandedNodes = []
        gameBoards=[]
        winner=0


        gameBoards.append(copy.deepcopy(self.board))
        isOffenseTurn = maxFirst
        self.currBoardIdx = self.startBoardIdx
        while self.checkMovesLeft() and self.checkWinner() == 0:
        # if True:
            self.bestMoveFound = None
            self.expandedNodes = 0
            print('---------------New Move-------------------')
            if isOffenseTurn:
                if isMinimaxOffensive:
                    value = self.minimax(0, self.currBoardIdx, isOffenseTurn)
                else:
                    value = self.alphabeta(0, self.currBoardIdx, -1E10, 1E10, isOffenseTurn)
            else:
                if isMinimaxDefensive:
                    value = self.minimax(0, self.currBoardIdx, isOffenseTurn)
                else:
                    value = self.alphabeta(0, self.currBoardIdx, -1E10, 1E10, isOffenseTurn)

            print(isOffenseTurn)
            # assert(self.minimax(0, self.currBoardIdx, isOffenseTurn) == self.alphabeta(0, self.currBoardIdx, -1E10, 1E10, isOffenseTurn))

            # Selected move
            # print('best found:', self.bestMoveFound)
            self.board[self.bestMoveFound[0]][self.bestMoveFound[1]] = self.maxPlayer if isOffenseTurn else self.minPlayer
            self.printGameBoard()

            bestMove.append(copy.deepcopy(self.bestMoveFound))
            bestValue.append(value)
            expandedNodes.append(self.expandedNodes)
            gameBoards.append(copy.deepcopy(self.board))

            # End turn
            isOffenseTurn = not isOffenseTurn
        print('-----------Final Board---------------')
        self.printGameBoard()
        print(self.checkMovesLeft())
        print(self.checkWinner())
        print(bestValue)
        print(expandedNodes)
        winner = self.checkWinner()
        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        bestValue=[]
        expandedNodes = []
        gameBoards=[]
        winner=0

        gameBoards.append(copy.deepcopy(self.board))
        #isAgentTurn = True if randint(0,1) == 0 else False
        isAgentTurn = False
        agentStart = isAgentTurn
        self.currBoardIdx = self.startBoardIdx
        print(self.currBoardIdx)
        print('Agent Start :', agentStart) 

        while self.checkMovesLeft() and self.checkWinner() == 0:
        # if True:
            self.bestMoveFound = None
            self.expandedNodes = 0
            print('---------------New Move-------------------')
            if isAgentTurn:
                print('Agent Move!')
                value = self.alphabeta_designed(0, self.currBoardIdx, -1E10, 1E10, isAgentTurn)
            else:
                value = self.alphabeta(0, self.currBoardIdx, -1E10, 1E10, isAgentTurn)

            #if agentStart :
            self.board[self.bestMoveFound[0]][self.bestMoveFound[1]] = self.maxPlayer if isAgentTurn else self.minPlayer
            #else :
             #   self.board[self.bestMoveFound[0]][self.bestMoveFound[1]] = self.minPlayer if isAgentTurn else self.maxPlayer

            self.printGameBoard()

            bestMove.append(copy.deepcopy(self.bestMoveFound))
            bestValue.append(value)
            expandedNodes.append(self.expandedNodes)
            gameBoards.append(copy.deepcopy(self.board))
            #print(bestValue)
            # End turn
            isAgentTurn = not isAgentTurn

        print('-----------Final Board---------------')
        print('Agent Start :', agentStart) 
        self.printGameBoard()
        print(bestValue)
        print(expandedNodes)
        winner = -self.checkWinner()

        print(winner)
        return gameBoards, bestMove, expandedNodes, bestValue, winner


    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        return gameBoards, bestMove, winner

if __name__=="__main__":
    uttt=ultimateTicTacToe()
    gameNum =  input('Game number? ') #str input
    predAgentWins = 0
    designAgentWins = 0

    #print(type(gameNum))

    #offensive first, all combinations
    #if gameNum == '1' :
    if gameNum == 1:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,True,True) #offensive(minimax) vs defensive(minimax)
        print("Game Number 1: offensive(minimax) vs defensive(minimax)")
    #elif gameNum == '2':
    elif gameNum == 2:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,True,False) #offensive(minimax) vs defensive(alpha-beta)
        print("Game Number 2: offensive(minimax) vs defensive(alpha-beta)")
    #elif gameNum == '3':
    elif gameNum == 3:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False,True) #offensive(alpha-beta) vs defensive(minimax)
        print("Game Number 3: offensive(alpha-beta) vs defensive(minimax)")
    #elif gameNum == '4' :
    elif gameNum == 4:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False,False) #offensive(alpha-beta) vs defensive(alpha-beta)
        print("Game Number 4: offensive(alpha-beta) vs defensive(alpha-beta)")
    #defensive first, all combinations

    if gameNum == 5:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,True,True) #offensive(minimax) vs defensive(minimax)
        print("Game Number 5: defensive(minimax) vs offensive(minimax)")
    #elif gameNum == '2':
    elif gameNum == 6:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,True,False) #offensive(minimax) vs defensive(alpha-beta)
        print("Game Number 6: defensive(minimax) vs offensive(alpha-beta)")
    #elif gameNum == '3':
    elif gameNum == 7:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,False,True) #offensive(alpha-beta) vs defensive(minimax)
        print("Game Number 7: defensive(alpha-beta) vs offensive(minimax)")
    #elif gameNum == '4' :
    elif gameNum == 8:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(False,False,False) #offensive(alpha-beta) vs defensive(alpha-beta)
        print("Game Number 8: defensive(alpha-beta) vs offensive(alpha-beta)")
    elif gameNum == 9:
        gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGameYourAgent()
    elif gameNum == 10:
        for i in range(20):
            uttt=ultimateTicTacToe()
            gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGameYourAgent()
            if winner == 1 :
                predAgentWins += 1
            elif winner == -1 :
                designAgentWins += 1
        
        print("Predefined agent wins: ", predAgentWins)
        print("Designed agent wins: ", designAgentWins)

    if gameNum < 9 :
        if winner == 1:
            print("The winner is maxPlayer!!!")
        elif winner == -1:
            print("The winner is minPlayer!!!")
        else:
            print("Tie. No winner:(")
    elif gameNum == 9:
        if winner == 1 :
            print("The winner is predefined Agent!!!")
        elif winner == -1:
            print("The winner is designed Agent!!!")
        else:
            print("Tie. No winner:(")
