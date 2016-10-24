import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q = defaultdict(float)
        self.alpa = 1.0
        self.gamma = 0.5
        self.actions = [None, 'forward', 'left', 'right']
        self.total_rewards = 0.0
        self.total_steps = 0

    def actionQ(self,state):
        ex = [self.q.get((state,action)) for action in self.actions]
        maxq = max(ex)
        if(self.q.get((state,self.next_waypoint),0.0) >= 0.0):
            act = self.actions.index(self.next_waypoint)
        elif(maxq >= 0.0):
            act = random.choice([k for k in range(len(self.actions)) if ex[k] >= 0.0])
        else:
            act = self.actions.index(None)
        return self.actions[act]

    def randomAction(self):
        return random.choice(self.actions)
    def learnQ(self,state,action,reward,nstate):
        prevq = self.q.get((state,action))
        exn = [self.q.get((nstate, a)) for a in self.actions]
        maxq = max(exn)
        if(prevq is None or maxq is None):
            self.q[(state, action)] = reward
        else:
            self.q[(state,action)] = prevq + self.alpa*(reward + self.gamma*maxq - prevq)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.q = defaultdict(float)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        
        # TODO: Select action according to your policy
        action = self.actionQ(self.state)
        #action = self.randomAction() # random action question1
        # Execute action and get reward
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward
        ninputs = self.env.sense(self)
        nstate = (ninputs['light'], ninputs['oncoming'], ninputs['left'], ninputs['right'])
        self.learnQ(self.state, action, reward, nstate)

        self.total_rewards += reward
        self.total_steps += 1
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, nextwaypoint={}, q={}".format(deadline, inputs, action, reward, self.next_waypoint,self.q)  # [debug]
        #print "LearningAgent.update(): total_steps so far = {}, total_rewards so far = {}".format(self.total_steps,self.total_rewards)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=1, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
