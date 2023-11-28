# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """ Search the deepest nodes in the search tree first. """

    currentState = problem.getStartState()
    stack = Stack()
    nodes_visited = [currentState]
    actions = []

    while not problem.isGoalState(currentState):

        # Get node's successors
        nextNodes = problem.getSuccessors(currentState)
        for i in nextNodes:
            # Check that successors haven't already been visited
            if i[0] not in nodes_visited:
                stack.push((i, currentState))

        # Get new node
        currentNode = stack.pop()
        currentState = currentNode[0][0]
        parentNode = currentNode[1]

        # Check that the currentNode's parent node is the last node on the path
        # if it's not then go back to parent node
        if (not (nodes_visited[len(nodes_visited)-1] == parentNode)):
            index_parent = nodes_visited.index(parentNode)
            nodes_visited = nodes_visited[:index_parent + 1]
            actions = actions[:index_parent]  # actions have an element less

        # Add node to path
        nodes_visited.append(currentState)
        actions.append(currentNode[0][1])

    return actions

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    currentState = problem.getStartState()
    queue = Queue()
    nodes_visited = [currentState]
    paths = [([currentState], [])]
    newAction = []

    while not problem.isGoalState(currentState):
        # Get node's successors
        nextNodes = problem.getSuccessors(currentState)
        for i in nextNodes:
            # Check that successors haven't already been visited
            if (i[0] not in nodes_visited):
                queue.push((i, currentState))
                nodes_visited.append(i[0])

        # Get new node
        if queue.isEmpty():
            break
        else:
            currentNode = queue.pop()
            currentState = currentNode[0][0]

        # Get path for the node
        for pathAction in paths:
            if (pathAction[0][len(pathAction[0])-1] == currentNode[1]):
                newPath = pathAction[0][:]
                newPath.append(currentState)
                newAction = pathAction[1].copy()
                newAction.append(currentNode[0][1])
                paths.append((newPath, newAction))
                break

    return newAction

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    currentState = problem.getStartState()
    queue = PriorityQueue()
    nodes_visited = set()
    nodes_visited.add(currentState)
    paths = [([currentState], [], 0)]

    while (not problem.isGoalState(currentState)):
        # Get node's successors
        nextNodes = problem.getSuccessors(currentState)
        for i in nextNodes:
            # Check that successors haven't already been visited
            if i[0] not in nodes_visited:
                for pathAction in paths:
                    if (pathAction[0][len(pathAction[0])-1] == currentState):
                        queue.update((i, currentState), i[2]+pathAction[2])
                        break

        # Get new node
        currentNode = queue.pop()
        currentState = currentNode[0][0]
        nodes_visited.add(currentState)

        # Get path for the node
        for pathAction in paths:
            if (pathAction[0][len(pathAction[0])-1] == currentNode[1]):
                newPath = pathAction[0][:]
                newPath.append(currentState)
                newAction = pathAction[1].copy()
                newAction.append(currentNode[0][1])
                cost = pathAction[2] + currentNode[0][2]
                paths.append((newPath, newAction, cost))
                break

    return newAction

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    currentState = problem.getStartState()
    queue = PriorityQueue()
    nodes_visited = set()
    nodes_visited.add(currentState)
    paths = [([currentState], [], 0)]

    while not problem.isGoalState(currentState):
        # Get node's successors
        nextNodes = problem.getSuccessors(currentState)
        for i in nextNodes:
            # Check that successors haven't already been visited
            if i[0] not in nodes_visited:
                for pathAction in paths:
                    if pathAction[0][len(pathAction[0])-1] == currentState:
                        queue.update((i, currentState), i[2]+pathAction[2] +
                                     heuristic(i[0], problem))
                        break

        # Get new node
        currentNode = queue.pop()
        currentState = currentNode[0][0]
        nodes_visited.add(currentState)

        # Get path for the node
        for pathAction in paths:
            if (pathAction[0][len(pathAction[0])-1] == currentNode[1]):
                newPath = pathAction[0][:]
                newPath.append(currentState)
                newAction = pathAction[1].copy()
                newAction.append(currentNode[0][1])
                cost = pathAction[2] + currentNode[0][2]
                paths.append((newPath, newAction, cost))
                break

    return newAction


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
