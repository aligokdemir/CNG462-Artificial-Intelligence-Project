import numpy as np
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

# Making the AI progress on several steps (n_step)

class NStepProgress:
    
    def __init__(self, environment, intelligence, n_steps):
        self.intelligence = intelligence
        self.rewards = []
        self.environment = environment
        self.n_steps = n_steps
    
    def __iter__(self):
        state = self.environment.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.intelligence(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.environment.step(action)
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            while len(history) > self.n_steps + 1:
                history.popleft()
            if len(history) == self.n_steps + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_steps + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state = self.environment.reset()
                history.clear()
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity  # capacity for array of memories
        self.n_steps = n_steps  # getting the result at each Nth step
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()  # buffer for moves

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        # randomly take the values from the array of moves to avoid local minima and maximas
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        # fill the memory with the experiences played before
        while samples > 0:
            entry = next(self.n_steps_iter) # 10 consecutive steps
            self.buffer.append(entry) # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
