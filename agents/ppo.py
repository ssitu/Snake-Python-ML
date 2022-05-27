import torch
# Constants
from torchinfo import summary

from agents.agent_pytorch import AgentPyTorch, calculate_discounted_rewards
from games.snake import Snake

ACTOR_LR = .0001
CRITIC_LR = .0001
DISCOUNT_FACTOR = 0.99
EPSILON = .1
# Times to train on the same experience
TRAININGS_PER_EPISODE = 5
ENTROPY_WEIGHT = .1

# To prevent NANs
SMALL_CONSTANT = .00001


class AgentPPO(AgentPyTorch):

    def __init__(self, snake_game: Snake, agent_name="AgentPPO", actor=None, critic=None):
        super().__init__(snake_game, agent_name=agent_name, actor=actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=ACTOR_LR)
        self.critic = critic
        if not self.critic:
            self.critic = self.construct_critic()
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=CRITIC_LR)
        self.critic_loss_fn = torch.nn.HuberLoss()
        self.states = []
        self.actions = []
        self.rewards = []
        summary(self.critic, (1, 1, self.snake_game.get_grid_height(),
                              self.snake_game.get_grid_width()))

    def construct_actor(self, obs_space, action_space) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LazyConv2d(50, 10),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.LazyConv2d(30, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, padding=1),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(30),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(action_space),
            torch.nn.Softmax(dim=-1)
        )

    def construct_critic(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LazyConv2d(50, 10),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.LazyConv2d(30, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, padding=1),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1),
        )

    def collect_experience(self, state, distribution, action):
        self.states.append(state)
        self.actions.append(action)

    def give_reward(self, reward: float):
        super().give_reward(reward)
        self.rewards.append(reward)

    def train_agent(self):
        states_batch = torch.cat(self.states, 0).detach()
        old_probabilities = self.actor(states_batch)
        # Isolate the probabilities of the performed actions under the new policy
        # And, detach since it is used as a constant, not for training
        old_action_probabilities = old_probabilities.gather(
            1, torch.tensor(self.actions).unsqueeze(1)).detach()
        for i in range(TRAININGS_PER_EPISODE):
            # L_clip = min( r * A, clip(r, 1-ep, 1+ep) * A )

            #
            # Calculate the ratio of the probability of the action under the new policy over the old
            #
            # Probabilities of the current policy
            new_probabilities = self.actor(states_batch)
            new_action_probabilities = new_probabilities.gather(
                1, torch.tensor(self.actions).unsqueeze(1))
            # Calculate ratios, r = p / p_old
            ratios = new_action_probabilities / old_action_probabilities
            # Clipped, clip(r, 1-ep, 1+ep)
            clipped_ratios = torch.clip(ratios, 1 - EPSILON, 1 + EPSILON)
            # Advantages, A = G - V(s)
            discounted_rewards = torch.tensor(calculate_discounted_rewards(
                self.rewards, DISCOUNT_FACTOR)).unsqueeze(1)
            state_values = self.critic(states_batch)
            advantages = discounted_rewards - state_values
            # Prevent the loss_clip from affecting the gradients of the critic
            advantages = advantages.detach()

            # Entropy
            entropy = ENTROPY_WEIGHT * - \
                (new_probabilities
                 * torch.log(torch.add(new_probabilities, SMALL_CONSTANT))).sum(dim=1)

            # Losses, L_clip, L_critic, L_entropy
            objective_clip = torch.min(
                ratios * advantages, clipped_ratios * advantages).mean()
            loss_critic = (self.critic_loss_fn(
                state_values, discounted_rewards)).mean()
            objective_entropy = entropy.mean()
            loss = -objective_clip + loss_critic - objective_entropy
            # Training
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self, filename=None):
        actor_filename = filename
        critic_filename = filename
        if filename:
            actor_filename += "Actor"
            critic_filename += "Critic"
        self.save_model(self.actor, self.actor_optimizer,
                        filename=actor_filename)
        self.save_model(self.critic, self.critic_optimizer,
                        filename=critic_filename)

    def load(self, filename=None):
        actor_filename = filename
        critic_filename = filename
        if filename:
            actor_filename += "Actor"
            critic_filename += "Critic"
        self.load_model(self.actor, self.actor_optimizer,
                        filename=actor_filename)
        self.load_model(self.critic, self.critic_optimizer,
                        filename=critic_filename)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        super().reset()
