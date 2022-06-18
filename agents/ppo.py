import numpy
import torch

from agents import utils
from agents.actor_critic_networks import ACNet
from agents.agent_torch import AgentTorch
from games.snake import Snake

# Constants
DISCOUNT_FACTOR = 0.99
EPSILON = .2
# Times to train on the same experience
EXTRA_TRAININGS_PER_EPISODE = 5
ENTROPY_WEIGHT = .01
# Small constant to prevent NANs
SMALL_CONSTANT = 1e-6


class AgentPPO(AgentTorch):

    def __init__(self, snake_game: Snake, model: ACNet, optimizer: torch.optim.Optimizer):
        super().__init__(snake_game)
        self.model = model
        self.optimizer = optimizer
        self.critic_loss_fn = torch.nn.HuberLoss()
        self.states = []
        self.actions = []
        self.rewards = []
        self.probabilities = []

    def get_action(self, obs: numpy.ndarray, training=True):
        state = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        # Sample action
        probabilities, _ = self.model(state)
        dist = torch.distributions.Categorical(probs=probabilities)
        action = dist.sample().item()
        if training:
            self.states.append(state)
            self.probabilities.append(probabilities)
            self.actions.append(action)
        return action

    def give_reward(self, reward: float):
        super().give_reward(reward)
        self.rewards.append(reward)

    def discounted_rewards(self):
        discounted_reward = 0.
        discounted_rewards = [0.] * len(self.rewards)
        for time_step, reward in zip(reversed(range(len(self.rewards))), reversed(self.rewards)):
            discounted_reward = DISCOUNT_FACTOR * discounted_reward + reward
            discounted_rewards[time_step] = discounted_reward
        return discounted_rewards

    def train_agent(self):
        states_batch = torch.cat(self.states).detach()
        old_probabilities = torch.cat(self.probabilities).detach()
        # Isolate the probabilities of the performed actions under the new policy
        old_action_probabilities = old_probabilities.gather(1, torch.tensor(self.actions).unsqueeze(1)).detach()

        # Do one iteration with the already calculated probabilities during the episode
        # In this case, ratio = 0
        for i in range(EXTRA_TRAININGS_PER_EPISODE):
            # L_clip = min(r(th)A, clip(r(th), 1 - ep, 1 + ep)A)

            #
            # Calculate the ratio of the probability of the action under the new policy over the old
            #

            # Probabilities of the current policy
            new_probabilities, state_values = self.model(states_batch)
            new_action_probabilities = new_probabilities.gather(1, torch.tensor(self.actions).unsqueeze(1))
            # Calculate ratios
            ratios = new_action_probabilities / old_action_probabilities

            # Clipped
            clipped_ratios = torch.clip(ratios, 1 - EPSILON, 1 + EPSILON)
            # Advantages
            discounted_rewards = torch.tensor(self.discounted_rewards()).unsqueeze(1)
            advantages = discounted_rewards - state_values
            advantages = advantages.detach()  # Prevent the loss_clip from affecting the gradients of the critic

            # Entropy
            entropy = ENTROPY_WEIGHT * -(
                    new_probabilities * torch.log(torch.add(new_probabilities, SMALL_CONSTANT))
            ).sum(dim=1)

            # Loss
            objective_clip = torch.min(ratios * advantages, clipped_ratios * advantages).sum()
            loss_critic = (self.critic_loss_fn(state_values, discounted_rewards)).sum()
            objective_entropy = entropy.sum()
            loss = -objective_clip + loss_critic - objective_entropy
            # Training
            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.optimizer.step()

    def save(self):
        utils.save(self.model, self.optimizer, self.model.name)

    def load(self):
        utils.load(self.model, self.optimizer, self.model.name)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.probabilities.clear()
        super().reset()
