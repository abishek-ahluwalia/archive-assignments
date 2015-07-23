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

import util

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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    #print problem.getSuccessors(problem.getSuccessors(problem.getStartState())[0][0])
       
     
    "*** YOUR CODE HERE ***"
    closed = []
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push ([problem.getStartState(), []  , 100000 ] , 100000 )
    while 1 :
	if fringe.isEmpty() : 
            return []
        node = fringe.pop()
        #print node[0],
        if problem.isGoalState(node[0]):
	    print "goal found"
           # print node
            return node[1]
        isPresent = 0
        for i in closed :
            if i == node[0] :
                isPresent = 1
        #print  node[0] , node[2]
        if isPresent == 0 :
            closed.append(node[0])
            successors = problem.getSuccessors(node[0])
            for i in successors :
                newNode = []
                path = []
                for j in node[1] :
                    path.append(j)
               # print i[1] , 
               # print path
		dup = 0  
                for j in closed :
                    if j == i[0]:
                        #print "duplicate",
                        dup = 1
                
                path.append(i[1])
                
                newNode.append(i[0] )
                newNode.append(path )
                newNode.append( node[2] - i[2])
                if dup == 0 :
                    fringe.push(newNode , newNode[2]) 
                 
        

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    closed = []
    from util import PriorityQueue
    from searchAgents import CornersProblem
    fringe = PriorityQueue()
    fringe.push ([problem.getStartState(), []  , 1 ] , 1)
    path = []
    cornersFound = 0
    while 1 :

	if fringe.isEmpty() : 
	    #print "emptyfringe"
            return []
        node = fringe.pop()
        #print node[0],

        if problem.isGoalState(node[0]):
	    #if not isinstance(  problem  , CornersProblem ) :
	    # print "goal found"
            return node[1]
            '''if len(problem.goals) == 0 :
		#print "recursion ended", 
		#print node ,
  		return node[1]

	    #print "goal found"
            #print node
	
	    problem.setStartState(node[0])
            node1 = breadthFirstSearch(problem)

	    for i in node1 :
		node[1].append(i)
            #print node[1]
 	    return node[1]
	    '''
            """for i in node[1] :
	            path.append(i)

            if not fringe.isEmpty() :
		fringe.pop()
            fringe.push([node[0] , [] , 1 ] , 1)
            if cornersFound == 1 :
	            return path
            """
        isPresent = 0
        for i in closed :
            if i == node[0] :
                isPresent = 1
        #print  node[0] , node[2] 
        if isPresent == 0 :
            closed.append(node[0])
            successors = problem.getSuccessors(node[0])
            for i in successors :
                newNode = []
                path = []
                for j in node[1] :
                    path.append(j)
               # print i[1] , 
               # print path
		dup = 0  
                for j in closed :
                    if j == i[0]:
                      #  print "duplicate",
                        dup = 1
                
                path.append(i[1])
                
                newNode.append(i[0] )
                newNode.append(path )
                newNode.append( node[2] + i[2])
                if dup == 0:
                    fringe.push(newNode , newNode[2]) 
                 
        

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    closed = []
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push ([problem.getStartState(), []  , 1 ] , 1)
    while 1 :
	if fringe.isEmpty() : 
            return []
        node = fringe.pop()
        #print node[0],
        if problem.isGoalState(node[0]):
	    print "goal found"
           # print node
            return node[1]
        isPresent = 0
        for i in closed :
            if i == node[0] :
                isPresent = 1
        #print  node[0] , node[2]
        if isPresent == 0 :
            closed.append(node[0])
            successors = problem.getSuccessors(node[0])
            for i in successors :
                newNode = []
                path = []
                for j in node[1] :
                    path.append(j)
               # print i[1] , 
               # print path
		dup = 0  
                for j in closed :
                    if j == i[0]:
                       # print "duplicate",
                        dup = 1
                
                path.append(i[1])
                
                newNode.append(i[0] )
                newNode.append(path )
                newNode.append( node[2] + i[2])
                if dup == 0 :
                    fringe.push(newNode , newNode[2]) 
                 
        

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = []
    from searchAgents import manhattanHeuristic
    from util import PriorityQueue
    from searchAgents import PositionSearchProblem
    fringe = PriorityQueue()
    fringe.push ([problem.getStartState(), []  , 1 ] , 1)
    while 1 :
	if fringe.isEmpty() : 
            return []
        node = fringe.pop()
        #print node[0][0], node[2]
        if problem.isGoalState(node[0]):
	    print "goal found"
            #print node
	    
            return node[1]
        isPresent = 0
        for i in closed :
            if i == node[0] :
                isPresent = 1
        #print  node[0] , node[2],
        if isPresent == 0 :
            closed.append(node[0])
            successors = problem.getSuccessors(node[0])
            for i in successors :
                newNode = []
                path = []
                for j in node[1] :
                    path.append(j)
               # print i[1] , 
               # print path
		dup = 0  
                for j in closed :
                    if j == i[0]:
                        #print "duplicate",
                        dup = 1
                
                path.append(i[1])
                
                newNode.append(i[0] )
                newNode.append(path )
                problem1 = problem 
                #if not isinstance(  problem  , PositionSearchProblem ) :
                #    for j in problem.goals :
                 #       problem1.goal = j
                newNode.append( node[2] + i[2] )
                if dup == 0 :
                    fringe.push(newNode , newNode[2]+ heuristic(i[0], problem )) 
                 
        

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
