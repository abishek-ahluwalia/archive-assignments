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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
	'''for i in range(newFood.width) :
		for j in range(newFood.height) :
		    if newFood[i][j] :
	'''		#print i,j 
        "*** YOUR CODE HERE ***"
	x,y = successorGameState.getPacmanPosition()
	GPos = [  successorGameState.getGhostPosition(i+1) for i in range(len(newGhostStates))]
	from util import manhattanDistance
	ghosts = [ manhattanDistance((x,y) , GPos[i])  for i in range(len(newGhostStates))]
	closestGhost = min(ghosts)
	if newGhostStates[0].scaredTimer > 0 and newGhostStates[0].scaredTimer > closestGhost:
		closestGhost = 100
	numFood = max(1 ,successorGameState.getNumFood())
	#from searchAgents import AnyFoodSearchProblem 
	#from search import bfs
	#foodb = max(1 , len(bfs(AnyFoodSearchProblem(successorGameState))))
	food = []
	for i in range(newFood.width) :
	    for j in range(newFood.height) :
		if newFood[i][j] :
		    food.append(manhattanDistance((x,y) , (i , j )))
	if len(food)==0 :
		food.append(1)
	foodm = max( 1 , min(food))
	#print foodm , foodb
        return (successorGameState.getScore() + 10/foodm+ closestGhost/10 + 100/numFood)

def scoreEvaluationFunction(currentGameState):
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
	self.currentDepth = self.depth
	self.statesExpanded = 0 

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction( self , state) :
	score , act = self.getValue(state , 0 , self.depth)
	#print self.statesExpanded , score
        return act

    def getValue(self , state , agentIndex , depth ) :
	if (depth ==0 ) or state.isWin() or state.isLose() :
	    return self.evaluationFunction( state) , None

	actions = state.getLegalActions(agentIndex)
	succ = [[] for i in range(len(actions))]
	val = [[] for i in range(len(actions))]
	act = [[] for i in range(len(actions))]
	for i in range( len(actions)):
	    action = actions[i]
	    succ[i] = state.generateSuccessor(agentIndex , action ) 
	    self.statesExpanded = self.statesExpanded+1
	    if ( agentIndex < state.getNumAgents() - 1 ) :
	        val[i] , act[i]  = self.getValue(succ[i] , agentIndex + 1 , depth)
            else :
	        val[i] , act[i]  = self.getValue(succ[i] , 0 , depth-1)
        if ( agentIndex == 0 ) : 
 	    return max( val )  , actions[val.index(max(val))]
	else : 
	    return min( val )  , actions[val.index(min(val))]
		
    def getActionOld(self, gameState):
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


	pacmanActions = gameState.getLegalActions(0)  
	#print gameState.getNumAgents(),
	#print "iter",self.currentDepth,
	#print pacmanActions ,
	#print gameState.isLose()
	#print gameState.isWin()
        succ = [[] for ag in range(gameState.getNumAgents())]	
	for i in  range(len(pacmanActions)) :
	    successor = gameState.generateSuccessor(0, pacmanActions[i]) 
	    #print successor.isWin()
	    if successor.isWin() : 
	
		succ[0].append([i , None , self.evaluationFunction(successor)])
		#print "win",self.evaluationFunction(successor),
		#print succ
	    else :
		#print succ
	        succ[0].append([i , successor,-999]) # first is the index of action , second is the successor for action , third is the score for action
	for j in  range( gameState.getNumAgents()-1 ) : 
	    next = []
	    for k in range(len(succ[j])) :
		if succ[j][k][1] :
		    if succ[j][k][1].isLose() :
			#print "lose",
			#print (succ)
			succ[j][k][2] = self.evaluationFunction(succ[j][k][1])
		    else :    
		        ghostActions = succ[j][k][1].getLegalActions(j+1)  
		        for act in ghostActions :
	   	            gSucc = succ[j][k][1].generateSuccessor(j+1, act) 
		            next.append([k , gSucc , 999])
	    succ[j+1] = next
	#print succ,
	if self.currentDepth >=1 :
		self.currentDepth = self.currentDepth -1
		'''print "reduce Depth",
		print ( self.currentDepth),'''
	for j in range(gameState.getNumAgents()):
	    ghos = gameState.getNumAgents()-1- j
	    for k in range(len(succ[ghos] )):
	        if j == 0 : 
		    if self.currentDepth == 0 or succ[ghos][k][1].isLose() or succ[ghos][k][1].isWin():
			#print succ[0]
			#print succ[1]
			#print "win1" , succ[ghos][k], succ[ghos][k][1].isWin() , 
			succ[ghos][k][2] = self.evaluationFunction(succ[ghos][k][1])
		    else :
			#print "call",
			#print ( self.currentDepth) ,
			succ[ghos][k][2] , act = self.getAction(succ[ghos][k][1])
			#print ( self.currentDepth) ,
			#print "end" , succ[ghos][k][0], act ,

   	        else:
		    nextScore = []
		    for l in range(len(succ[ghos +1])) :
			if k == succ[ghos+ 1][l][0] :
		 	    nextScore.append(succ[ghos+1][l][2])
			    #print "ns" , nextScore, succ[ghos][k][0],
		    if ( len(nextScore) > 0 ) :
			    #print j , succ[ghos][k][0] , min(nextScore), "min",
			    succ[ghos][k][2] = min(nextScore)
	maxScore = -999
	maxScoreIndex = 0 
	for i in range(len( pacmanActions)) :
	    ''' print "max",
	    print ( i ),
	    print ( pacmanActions),
	    print succ[0][i][2],
	    #print succ'''
	    maxScore = max(succ[0][i][2] , maxScore) 
	    if maxScore ==  succ[0][i][2]:
	        maxScoreIndex = i 
	if self.currentDepth != self.depth-1 and self.depth !=1 :
	    '''print "ret",
	    print (self.currentDepth),
	    #print (self.depth),
	    print pacmanActions[maxScoreIndex],
	    print maxScore ,'''
	    self.currentDepth = self.currentDepth+1
            return maxScore , pacmanActions[maxScoreIndex]
	else :
	    print "ret over",maxScore
	    print pacmanActions[maxScoreIndex],
	    self.currentDepth = self.currentDepth+1
	    return pacmanActions[maxScoreIndex]
	'''
	else :
	    pacmanActions = gameState.getLegalActions(0)  
	    for i in  pacmanActions :
		successor = gameState.generateSuccessor(agentIndex, action) 
		prev = [[] for ag in gameState.getNumAgents()]
	        prev[0].append(successor)
		for j in  range(  gameState.getNumAgents()-1 ) : 
		    next = []
		    for k in prev[j] :
			ghostActions = k.getLegalActions(j+1)  
			for act in ghostActions :
	   		    gSucc = gameState.generateSuccessor(j+1, act) 
			    next.append(gSucc)
			    prev.remove(k)
		    prev[j+1] = next
		for suc in prev[gameState.getNumAgents()-1]
		    self.getAction(suc , self.depth-1)
	'''
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
	
        "*** YOUR CODE HERE ***"
	alpha = -99999999
	beta = 99999999
	val , act = self.getValue ( gameState , 0 , self.depth , alpha , beta ) 
	#print self.statesExpanded , "ab" , val
	return act 
	
	
	
        util.raiseNotDefined()


    def getValue( self , state , agentIndex , depth , alpha , beta ) :
	 
	if  depth == 0  or state.isWin() or state.isLose() :
		return self.evaluationFunction(state) , None
	    


	actions = state.getLegalActions(agentIndex)
	succ = [[] for i in range(len(actions))]
	val = [[] for i in range(len(actions))]
	act = [[] for i in range(len(actions))]
	for i in range(len(actions)) : 
	    action = actions[i]
	    succ[i] = state.generateSuccessor(agentIndex , action) 
	    self.statesExpanded = self.statesExpanded + 1
	    if agentIndex < state.getNumAgents() - 1 : 
	        val[i] , act[i] = self.getValue(succ[i] , agentIndex + 1 , depth, alpha , beta)
            else: 
		val[i] , act[i] = self.getValue(succ[i] , 0 , depth -1 , alpha , beta ) 
	    if agentIndex == 0 and val[i] > alpha :
                #print "alpha change " , val[i]  , alpha , beta
		alpha = val[i] 
	    if agentIndex == 0 and val[i] > beta  :
		#print "beta " , val[i]  , alpha , beta 
		return val[i] , act[i] 

	    if agentIndex != 0  and val[i] < beta : 
	 	#print "beta change" , val[i]  , alpha , beta
		beta = val[i] 
	    if agentIndex != 0 and val[i] < alpha :
		#print "alpha " , val[i]  , alpha , beta 
		return val[i] , act[i]
        if agentIndex == 0 : 
	    #print "pac" , val  , alpha , beta 
	    return max(val) ,  actions[val.index(max(val))]
	else :
	    #print "min" , val  , alpha , beta 
	    return min(val) ,  actions[val.index(min(val))]

	

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def getValue(self , state , agentIndex , depth ) :
	if (depth ==0 ) or state.isWin() or state.isLose() :
	    return self.evaluationFunction( state) , None

	actions = state.getLegalActions(agentIndex)
	succ = [[] for i in range(len(actions))]
	val = [[] for i in range(len(actions))]
	act = [[] for i in range(len(actions))]
	for i in range( len(actions)):
	    action = actions[i]
	    succ[i] = state.generateSuccessor(agentIndex , action ) 
	    self.statesExpanded = self.statesExpanded+1
	    if ( agentIndex < state.getNumAgents() - 1 ) :
	        val[i] , act[i]  = self.getValue(succ[i] , agentIndex + 1 , depth)
            else :
	        val[i] , act[i]  = self.getValue(succ[i] , 0 , depth-1)
        if ( agentIndex == 0 ) : 
 	    return max( val )  , actions[val.index(max(val))]
	else : 
	    return sum( val )/len(val)  , None
		
    def getAction(self, state):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
	score , act = self.getValue(state , 0 , self.depth)
	return act 
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
        """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
        """
        "*** YOUR CODE HERE ***"
	newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	'''for i in range(newFood.width) :
		for j in range(newFood.height) :
		    if newFood[i][j] :
	'''		#print i,j 
        "*** YOUR CODE HERE ***"
	x,y = currentGameState.getPacmanPosition()
	GPos = [  currentGameState.getGhostPosition(i+1) for i in range(len(newGhostStates))]
	from util import manhattanDistance
	ghosts = [ manhattanDistance((x,y) , GPos[i])  for i in range(len(newGhostStates))]
	closestGhost = min(ghosts)
	if newGhostStates[0].scaredTimer > 0 and newGhostStates[0].scaredTimer > closestGhost:
		closestGhost = 100
	numFood = max(1 ,currentGameState.getNumFood())
	#from searchAgents import AnyFoodSearchProblem 
	#from search import bfs
	#foodb = max(1 , len(bfs(AnyFoodSearchProblem(successorGameState))))
	food = []
	for i in range(newFood.width) :
	    for j in range(newFood.height) :
		if newFood[i][j] :
		    food.append(manhattanDistance((x,y) , (i , j )))
	if len(food)==0 :
		food.append(1)
	foodm = max( 1 , min(food))
	#print foodm , foodb
        return (currentGameState.getScore() + 10/foodm+ closestGhost/10 + 100/numFood)
        util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

