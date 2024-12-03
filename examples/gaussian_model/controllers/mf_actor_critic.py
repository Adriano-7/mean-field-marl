from examples.gaussian_model.controllers.actor_critic import ActorCritic
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class MFActorCriticNet(nn.Module):
    def __init__(self, input_shape, nr_actions, nr_hidden_units):
        super(MFActorCriticNet, self).__init__()
        self.nr_input_features = numpy.prod(input_shape)
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )
        self.action_head = nn.Linear(self.nr_hidden_units, nr_actions)
        self.value_head = nn.Linear(self.nr_hidden_units, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)

class MFAC(ActorCritic):
    def __init__(self, params):
        super(MFAC, self).__init__(params)
        self.policy_net = MFActorCriticNet(self.local_observation_space, self.nr_actions, params["actor_hidden_units"])
        self.parameters = self.policy_net.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def local_probs(self, observations, training_mode=True):
        observations = observations.view(-1, self.policy_net.nr_input_features)
        probs, _ = self.policy_net(observations)
        return probs.detach()

    def centralized_update(self, states, joint_actions, observations, old_probs, dones, returns, subteam_indices):
        result = (0, 0)
        if self.critic_learner is not None:
            for _ in range(self.nr_update_iterations):
                result = self.critic_learner.update(states, joint_actions, observations, returns, dones, old_probs, subteam_indices)
        return result

    def local_update(self, states, joint_actions, observations, old_probs, dones, returns):
        batch_size = states.size(0)
        for _ in range(self.nr_update_iterations):
            action_probs, advantages, values = self.action_probs_and_advantages(states, joint_actions, observations, returns)
            action_probs = action_probs.view(batch_size, self.nr_agents, self.nr_actions)
            advantages = advantages.view(batch_size, self.nr_agents, 1)
            values = values.view(batch_size, self.nr_agents)
            policy_losses = []
            value_losses = []
            for joint_action, old_joint_probs, joint_probs, joint_advantages, joint_R, joint_value in zip(joint_actions, old_probs, action_probs, advantages, returns, values):
                for action, old_prob, probs, advantage, R, value in zip(joint_action, old_joint_probs, joint_probs, joint_advantages, joint_R, joint_value):
                    if probs.sum() > self.eps:
                        policy_losses.append(self.policy_loss(advantage.item(), probs, action, old_prob))
                        value_losses.append(F.mse_loss(R, value))
            loss = torch.stack(policy_losses).mean()
            if self.critic_learner is None:
                loss += torch.stack(value_losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
            self.optimizer.step()
        return True