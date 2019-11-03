import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.action_size = action_size
		self.gamma = .95
		self.epsilon = .99
		self.epsilon_min = .01
		self.epsilon_decay = .995
		self.learning_rate = .001
		self.memory = deque(maxlen=2000)
		self.state_size = state_size
		self.model = self._build_model()

	def act(self, state):
		if np.random.rand() < self.epsilon:
			return random.randrange(self.action_size)
		else:
			return np.argmax(self.model.predict(state)[0])

	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))

		model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def train_on_memory(self, batch_size):
		# The goal is to predict the reward depending on the action
		batch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in batch:
			target_f = self.model.predict(state)
			if done:
				target_f[0][action] = reward
			else:
				target_f[0][action] = reward * self.gamma * np.amax(self.model.predict(next_state)[0])
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load_weights(self, name):
		self.model.load_weights(name)

	def save_weights(self, name):
		self.model.save_weights(name)

if __name__ == "__main__":
	env = gym.make('CartPole-v1')

	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	agent = DQNAgent(state_size, action_size)
	agent.load_weights("./save/cartpole-dqn.h5")

	EPISODES = 500
	BATCH_SIZE = 32

	for e in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])

		frame = 0
		done = False

		while not done:
			env.render()
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			frame = frame + 1

			if len(agent.memory) > BATCH_SIZE:
				agent.train_on_memory(BATCH_SIZE)

		print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, frame, agent.epsilon))

		if e % 10 == 0:
			agent.save_weights("./save/cartpole-dqn.h5")
