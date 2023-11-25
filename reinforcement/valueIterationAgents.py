# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()  

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations > 0:
            self.iterations -= 1
            updated_values = {}
            states_to_update = set()

            for state in self.mdp.getStates():
                optimal_action = self.computeActionFromValues(state)
                if optimal_action:
                    states_to_update.add(state)
                    updated_values[state] = self.computeQValueFromValues(state, optimal_action)

            for state in states_to_update:
                self.values[state] = updated_values[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total_value = 0
        transition_details = self.mdp.getTransitionStatesAndProbs(state, action)

        for next_state, probability in transition_details:
            reward = self.mdp.getReward(state, action, next_state)
            total_value += probability * (reward + self.discount * self.getValue(next_state))

        return total_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        available_actions = self.mdp.getPossibleActions(state)
    
        if not available_actions:
            return None

        best_action = None
        best_reward = float('-inf')
    
        for action in available_actions:
            transition_details = self.mdp.getTransitionStatesAndProbs(state, action)
            current_reward = sum(prob * (self.mdp.getReward(state, action, next_state) + 
                            self.discount * self.getValue(next_state)) 
                            for next_state, prob in transition_details)
            
            if current_reward > best_reward:
                best_reward = current_reward
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        adjacency_list = []
        index_mapping = util.Counter()
        index_counter = 0

        for source_state in all_states:
            neighboring_states = set()
            for target_state in all_states:
                possible_actions = self.mdp.getPossibleActions(target_state)
                for current_action in possible_actions:
                    transition_probs = self.mdp.getTransitionStatesAndProbs(target_state, current_action)
                    for transition_state, transition_prob in transition_probs:
                        if transition_state == source_state and transition_prob > 0:
                            neighboring_states.add(target_state)
            adjacency_list.append(neighboring_states)
            index_mapping[source_state] = index_counter
            index_counter += 1

        priority_queue = util.PriorityQueue()
        updated_values = util.Counter()

        for state in all_states:
            action_options = self.mdp.getPossibleActions(state)
            if self.mdp.isTerminal(state):
                continue
            previous_value = self.getValue(state)
            optimal_action = self.computeActionFromValues(state)
            if optimal_action:
                updated_val = self.computeQValueFromValues(state, optimal_action)
                updated_values[state] = updated_val
                value_difference = abs(previous_value - updated_val)
                priority_queue.push(state, -value_difference)
            else:
                updated_values[state] = previous_value

        for _ in range(self.iterations):
            if priority_queue.isEmpty():
                break
            top_state = priority_queue.pop()
            if not self.mdp.isTerminal(top_state):
                self.values[top_state] = updated_values[top_state]
            for predecessor in adjacency_list[index_mapping[top_state]]:
                prev_val = self.getValue(predecessor)
                optimal_act = self.computeActionFromValues(predecessor)
                if optimal_act:
                    updated_val_pred = self.computeQValueFromValues(predecessor, optimal_act)
                    value_diff_pred = abs(prev_val - updated_val_pred)
                    updated_values[predecessor] = updated_val_pred
                    if value_diff_pred > self.theta:
                        priority_queue.update(predecessor, -value_diff_pred)
