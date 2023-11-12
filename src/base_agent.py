
# all agents must follow the same design paradigm
class BaseAgent:
    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps):
        raise NotImplementedError
