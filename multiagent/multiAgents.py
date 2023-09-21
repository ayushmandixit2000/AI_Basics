# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        closestGhost = float('inf')
        foods = newFood.asList()
        for ghost in newGhostStates:
            x, y = ghost.getPosition()
            x = int(x)
            y = int(y)
            if ghost.scaredTimer == 0:
                closestGhost = min(closestGhost,manhattanDistance((x, y), newPos))
        closestFood = float('inf')
        for food in foods:
            closestFood = min(closestFood, manhattanDistance(food, newPos))
        if not foods:
            closestFood = 0

        return successorGameState.getScore() - 5 / (closestGhost + 1)  - closestFood / 4 #modified reciprocal


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        return self.value(gameState, self.depth, 0)[1]

    def value(self, gameState, depth, agent):
        if depth == 0 or gameState.isLose() or gameState.isWin(): #terminal state, return state's utility
            return self.evaluationFunction(gameState), Directions.STOP

        elif agent == 0: #next agent is MAX
            return self.maxValue(gameState, depth, agent)

        else: #next agent is MIN
            return self.minValue(gameState, depth, agent)

    def maxValue(self, gameState, depth, agent):
        agentActions = gameState.getLegalActions(agent)

        if agent == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1

        else:
            nextAgent, nextDepth = agent + 1, depth

        maxScore = float('-inf')
        maxAction = Directions.STOP

        for action in agentActions:
            successorState = gameState.generateSuccessor(agent, action)
            newScore = self.value(successorState, nextDepth, nextAgent)[0]
            if newScore > maxScore:
                maxScore, maxAction = newScore, action
        return maxScore, maxAction

    def minValue(self, gameState, depth, agent):

        agentActions = gameState.getLegalActions(agent)

        if agent == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1

        else:
            nextAgent, nextDepth = agent + 1, depth

        minScore = float('inf')
        minAction = Directions.STOP

        for action in agentActions:
            successorState = gameState.generateSuccessor(agent, action)
            newScore = self.value(successorState, nextDepth, nextAgent)[0]
            if newScore < minScore:
                minScore, minAction = newScore, action
        return minScore, minAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, self.depth, 0, float('-inf'), float('inf'))[1]

    def value(self, gameState, depth, agent, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        elif agent == 0: #next agent is MAX
            return self.maxValue(gameState, depth, agent, alpha, beta)

        else: #next agent is min
            return self.minValue(gameState,depth, agent, alpha, beta)
        

    def maxValue(self, gameState, depth, agent, alpha, beta):
        legalActions = gameState.getLegalActions(agent)

        if agent == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1

        else:
            nextAgent, nextDepth = agent + 1, depth

        maxScore, maxAction = float('-inf'), Directions.STOP

        for action in legalActions:
            successorState = gameState.generateSuccessor(agent, action)
            newScore = self.value(successorState, nextDepth, nextAgent ,alpha, beta)[0]

            if newScore > maxScore:
                maxScore, maxAction = newScore, action
            if newScore > beta:
                return newScore, action
            alpha = max(alpha, maxScore)
        return maxScore, maxAction

    def minValue(self, gameState, depth, agent, alpha, beta):
        legalActions = gameState.getLegalActions(agent)

        if agent == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1

        else:
            nextAgent, nextDepth = agent + 1, depth
        
        minScore, minAction = float('inf'), Directions.STOP
        
        for action in legalActions:
            successorState = gameState.generateSuccessor(agent, action)
            newScore = self.value(successorState, nextDepth, nextAgent ,alpha, beta)[0]

            if newScore < minScore:
                minScore, minAction = newScore, action
            if newScore < alpha:
                return newScore, action
            beta = min(beta, minScore)

        return minScore, minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, self.depth, 0)[1]

    def value(self, gameState, depth, agent):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        elif agent == 0: #next agent is MAX
            return self.maxValue(gameState, depth, agent)
        else:
            return self.expValue(gameState, depth, agent)

    def maxValue(self, gameState, depth, agent):
        agentActions = gameState.getLegalActions(agent)

        if agent == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1

        else:
            nextAgent, nextDepth = agent + 1, depth

        maxScore = float('-inf')
        maxAction = Directions.STOP

        for action in agentActions:
            successorState = gameState.generateSuccessor(agent, action)
            newScore = self.value(successorState, nextDepth, nextAgent)[0]
            if newScore > maxScore:
                maxScore, maxAction = newScore, action
        return maxScore, maxAction

    def expValue(self, gameState, depth, agent):
        agentActions = gameState.getLegalActions(agent)

        if agent == gameState.getNumAgents() - 1:
            nextAgent, nextDepth = 0, depth - 1

        else:
            nextAgent, nextDepth = agent + 1, depth

        expScore, expAction = 0, Directions.STOP
        for action in agentActions:
            successorState = gameState.generateSuccessor(agent, action)
            expScore += self.value(successorState, nextDepth, nextAgent)[0]
        expScore /= len(agentActions)
        return expScore, expAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foods = food.asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    closestGhost = float('-inf')

    for ghostState in ghostStates:
        x, y = ghostState.getPosition()
        x = int(x)
        y = int(y)
        if ghostState.scaredTimer == 0:
            closestGhost = min(closestGhost,manhattanDistance((x, y), pacmanPos))
        else:
            closestGhost = -100

    closestFood = float('inf')

    for food in foods:
        closestFood = min(closestFood, manhattanDistance(food, pacmanPos))

    if not foods:
        closestFood = 0

    return currentGameState.getScore()-5/(closestGhost+1) - closestFood/3

# Abbreviation
better = betterEvaluationFunction
