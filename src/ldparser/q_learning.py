import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import src.setup as setup
import json
from enum import Enum
from os.path import exists

class DeepQNetwork(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_outputs):
		super(DeepQNetwork, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_outputs = n_outputs
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.fc3 = nn.Linear(self.fc2_dims, self.n_outputs)  
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
		self.to(self.device)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		actions = self.fc3(x)

		return actions

class Agent():
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_outputs, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_min = eps_end
		self.eps_dec = eps_dec
		self.lr = lr
		self.action_space = [i for i in range(n_outputs)]
		self.mem_size = max_mem_size
		self.batch_size = batch_size
		self.mem_counter = 0

		self.Q_eval = DeepQNetwork(lr, n_outputs=n_outputs, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)

	def store_transition(self, state, action, reward):
		index = self.mem_counter % self.mem_size
		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward

		self.mem_counter += 1


	def calculate_actions(self, steer_profile):
		results = {}
		for candidate_action in setup.Actions:
			state = T.tensor(np.array(steer_profile, dtype=np.float32)).to(self.Q_eval.device)
			action_eval = self.Q_eval.forward(state)
			results[candidate_action] = action_eval[candidate_action].item()

		return results

	def learn(self):
		if self.mem_counter < self.batch_size:
			return

		print("learning")
		self.Q_eval.optimizer.zero_grad()

		max_mem = min(self.mem_counter, self.mem_size)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
		action_batch = self.action_memory[batch]

		q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
		q_target = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)

		loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backward()
		self.Q_eval.optimizer.step()

		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

	def write_training_input(self, steer_profile, time, action, file_name="./training_input.json"):
		output = {
			"steer_profile" : steer_profile,
			"time" : time,
			"action": action
		}
		with open(file_name, 'w') as f:
			json.dump(output, f, indent=4, separators=(',', ': '))

	def read_training_input(self, file_name="./training_input.json"):
		with open(file_name) as f:
			self.input = json.load(f)
		
		return self.input


	def mainloop():
		agent = Agent(gamma=0.99, epsilon = 1.0, batch_size = 64, n_outputs = 1, eps_end = 0.01,
			input_dims=[10], lr=0.03)
		while True:
			action = agent.choose_action(observation)
			observation_, reward = getResult()
			agent.store_transition(observation, action, reward)
			agent.learn()

	def calc_reward(self, last_profile, last_time, current_profile, current_time, target_time):
		old_square_profile_sum = np.sum(last_profile)
		new_square_profile_sum = np.sum(current_profile)
		profile_diff = old_square_profile_sum - new_square_profile_sum

		time_diff = (last_time - current_time) * np.exp(target_time-current_time)

		return profile_diff + time_diff

	def save(self, path="./data/nn_data"):
		T.save(self.Q_eval.state_dict(), path+".pt")
		np.save(path+"_mem_counter.npy", [self.mem_counter])
		np.save(path+"_state_memory.npy", self.state_memory)
		np.save(path+"_action_memory.npy", self.action_memory)
		np.save(path+"_reward_memory.npy", self.reward_memory)

       
	def load(self, path="./data/nn_data"):
		if exists(path+".pt"):
			self.Q_eval.load_state_dict(T.load(path+".pt"))
			self.Q_eval.eval()

			self.mem_counter = np.load(path+"_mem_counter.npy", allow_pickle=True)[0]
			self.state_memory = np.load(path+"_state_memory.npy", allow_pickle=True)
			self.action_memory = np.load(path+"_action_memory.npy", allow_pickle=True)
			self.reward_memory = np.load(path+"_reward_memory.npy", allow_pickle=True)



