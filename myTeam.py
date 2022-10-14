# myTeam.py
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


import sys
from captureAgents import CaptureAgent
import random, time, util
import distanceCalculator
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CarefulOffenseAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# Base reflex agent logic from provided code, subclass it
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    # Use to track agent action history
    self.actionTracker = ["STOP"]
    CaptureAgent.registerInitialState(self, gameState)
    # Use for keeping track of which coordinates are points of interest
    self.isRed = self.start[0] < 25

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    
    # Observe each action our agent takes
    if self.index == 1:
      print(values, file=sys.stderr)
      # print(self.getPreviousObservation(), file=sys.stderr)

    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    
    # Find the max rated action
    maxValue = max(values)
    # Generate a list of all actions that match the maxValue
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    # if self.index == 1:
    #   print(bestActions, file=sys.stderr)

    foodLeft = len(self.getFood(gameState).asList())
    # If total food left to collect is less than 2 or agent is carrying 2+
    if foodLeft <= 2 or gameState.getAgentState(self.index).numCarrying > 2:
      # Initialize for use in choosing next action
      bestDist = 9999
      # Loop through each legal action for the agent
      for action in actions:
        # Look at the future gamestate for said action
        successor = self.getSuccessor(gameState, action)
        # Grab the agent position for that future state
        pos2 = successor.getAgentPosition(self.index)
        # Find the distance between that pos and agent start pos
        dist = self.getMazeDistance(self.start,pos2)
        # Check if the action makes agent closer to it's start pos
        if dist < bestDist:
          # Then this is the best action so far to get to safety
          bestAction = action
          # Use for next action comparison
          bestDist = dist
      # Return action that gets agent closest to its start po
      return bestAction
    
    bestAction = random.choice(bestActions)
    # Check if action is stop, if so choose something else randomly
    # Rough solution to the cowering problem
    if bestAction == 'Stop':
      actions.remove('Stop')
      bestAction = random.choice(actions)
    
    return bestAction

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    # Store the future gamestate after specified action is taken
    successor = gameState.generateSuccessor(self.index, action)
    # New pos of agent in future gamestate
    pos = successor.getAgentState(self.index).getPosition()
    
    if pos != nearestPoint(pos):
      # Only half a grid position was covered, so advance state again?
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    # Counter of game score for each successor gamestate after action
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    # if self.index == 1:
    #   print(str(features) + str(weights), file=sys.stderr)
    #   # print(gameState.getAgentState(self.index)) # Print out a text representation of the world.

    return features * weights

  # Edit this to change behavior by changing what determines a feature score
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    # Dictionary with key: int pairs
    features = util.Counter()
    # Look at next gamestate
    successor = self.getSuccessor(gameState, action)
    # Calculate the score differential for that gamestate and store
    features['successorScore'] = self.getScore(successor)
    # Returns how much team is winning or losing after action
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    # Used to weight the value of features in evaluation function
    return {'successorScore': 1.0}

class CarefulOffenseAgent(ReflexCaptureAgent):
  
  # Create a rating for features of the gamestate, like score, food dist, flee
  def getFeatures(self, gameState, action):
    # Keep track of feature values
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    # State of agent after action is taken
    myState = successor.getAgentState(self.index)
    # Agent position after action
    myPos = myState.getPosition()

    if self.getScore(successor) > 5:
      # if team is winning by decent amount then it switches to defensive agent
      # See if agent is on home side
      if myPos[0] == 30:
        features['inHome'] = 1

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      # Generate list of enemy pacman that we can see
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      # Add number of invaders as a feature
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        # Make list of distances of visible enemy invaders
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        # Add closest distance enemy as feature
        features['invaderDistance'] = min(dists)
      else:
        # Set points of interest for defense based on which team we are
        if self.isRed:
          pointsOfInterest = [(10, 3), (12, 6), (12, 12)]
          distsFromPoints = [self.getMazeDistance(myPos, p) for p in pointsOfInterest]
          features['stayNearPOI'] = max(distsFromPoints)
        else:
          pointsOfInterest = [(20, 11), (18, 7), (18, 3)]
          distsFromPoints = [self.getMazeDistance(myPos, p) for p in pointsOfInterest]
          features['stayNearPOI'] = max(distsFromPoints)
      
      if action == Directions.STOP: features['stop'] = 1
      # Calculate the opposite direction of current action and add feature
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

    # if team is not winning by enough it uses this offense agent
    else:
      # List of edible food
      foodList = self.getFood(successor).asList() 
      # Set a feature, successorScore, as the negation of remaining food   
      features['successorScore'] = -len(foodList)

      # Compute distance to the nearest food
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        # Store the distance of the closest food
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        # Add this distance as a feature
        features['distanceToFood'] = minDistance

      # Determine if the enemy is closer to you than they were last time
      # and you are in their territory.
      # Note: This behavior isn't perfect, and can force Pacman to cower 
      # in a corner.  I leave it up to you to improve this behavior.
      

      distToHome = self.getMazeDistance(self.start, gameState.getAgentState(self.index).getPosition())

      enemy_dist = 9999.0
      entrance_dist = 9999.0
      # If our agent is in Pacman form
      if gameState.getAgentState(self.index).isPacman:
        # Generate a list of enemies future states after action is taken
        opp_fut_state = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        

        # print("TEST-------------------")
        # print(gameState.getLegalActions(([i for i in self.getOpponents(successor)][0])))

        # Grab the enemies that are currently ghosts and are alive
        chasers = [p for p in opp_fut_state if p.getPosition() != None and not p.isPacman]
        # If there are potential enemy chasers
        if len(chasers) > 0:
          # Find the closest distance chaser to the agent
          enemy_dist = min([float(self.getMazeDistance(myPos, c.getPosition())) for c in chasers])
      
      # IDEA: simulate some potential enemy paths, and avoid
      # If our action is on a potential enemy path
        # If enemy gets further away after action, increase feature
        # If closer, decrease feature
      
      # NOTE: enemy state is hidden from us if they are outside agent "sight" range, line 290 in capture.py
      else:
        opp_fut_state = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        chasers = [p for p in opp_fut_state if p.getPosition() != None and not p.isPacman]
        if len(chasers) > 0:
          if self.isRed:
            entrances = [(10, 3), (12, 6), (12, 12)]
          else:
            entrances = [(20, 14), (20, 6), (20, 2)]
          entrance_dist = max([float(self.getMazeDistance(myPos, e)) for e in entrances])

      # A
      features['furthestEntrance'] = entrance_dist
      # Store the feature under fleeEnemy as the inverse of enemy_dist
      features['fleeEnemy'] = 1.0/enemy_dist
    
    return features
  
  # Change how different features should impact chosen actions
  def getWeights(self, gameState, action):
    # Score is weighted high, food distance middle, and fleeEnemy high.
    # Negatives mean that actions that bring an agent higher values for those items 
    # are weighted less, decreasing likelihood of choosing said action. For ex,
    # actions that increase fleeEnemy mean we are getting closer to an enemy,
    # so we shouldn't choose that action as often. When total food is running low,
    # agent is less likely to choose food-pursuing actions.
    successor = self.getSuccessor(gameState, action)
    if self.getScore(successor) > 5:
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -1, 'stop': -100, 'reverse': -2, 'stayNearPOI': -0.5, 'inHome': -100}
    else:
      return {'successorScore': 100, 'distanceToFood': -1, 'fleeEnemy': -100.0, 'furthestEntrance': -100}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # See if agent is on home side
    if myPos[0] == 30:
      features['inHome'] = 1

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    # Generate list of enemy pacman that we can see
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    # Add number of invaders as a feature
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      # Make list of distances of visible enemy invaders
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # Add closest distance enemy as feature
      features['invaderDistance'] = min(dists)
    else:
      # Set points of interest for defense based on which team we are
      if self.isRed:
        pointsOfInterest = [(10, 3), (12, 6), (12, 12)]
        distsFromPoints = [self.getMazeDistance(myPos, p) for p in pointsOfInterest]
        features['stayNearPOI'] = min(distsFromPoints)
      else:
        pointsOfInterest = [(20, 11), (18, 7), (18, 3)]
        distsFromPoints = [self.getMazeDistance(myPos, p) for p in pointsOfInterest]
        features['stayNearPOI'] = min(distsFromPoints)
    
    if action == Directions.STOP: features['stop'] = 1
    # Calculate the opposite direction of current action and add feature
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -1, 'stop': -100, 'reverse': -2, 'stayNearPOI': -0.5, 'inHome': -100}
