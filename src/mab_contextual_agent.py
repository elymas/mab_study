import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


class LinUCB:
    def __init__(self, d, n, alpha):
        if d <=0 or n <=0:
            raise ValueError('Number of features and arms must be > 0')

        self.A_a = [np.eye(d) for _ in range(n)]
        self.b_a = [np.zeros((1, d)) for _ in range(n)]
        self.theta_hat_a = [np.zeros((d, 1)) for _ in range(n)]
        self.alpha = alpha
        self.d = d
        self.n = n

    def receive_reward(self, context, arm, reward):
        self.receive_rewards([context], [arm], [reward])

    def receive_rewards(self, contexts, arms, rewards):
        if len(contexts) != len(arms) or len(contexts) != len(rewards):
            raise ValueError('Must give the same number of contexts, arms, and rewards')

        for i in range(len(contexts)):
            if len(contexts[i]) != self.d:
                raise ValueError('Context does not have the same dimentions as given in the constructor')

            x = np.array(contexts[i]).reshape((-1, 1)) # 1차원 배열을 열벡터로 변환 (n,) -> (n,1)
            x_t = x.transpose()

            xMultx_t = np.dot(x, x_t)
            contextMultipliedWithReward = np.multiply(contexts[i], rewards[i])

            cur_arm = arms[i]
            self.A_a[cur_arm] += xMultx_t
            self.b_a[cur_arm] += contextMultipliedWithReward.reshape((1, -1))

            self.theta_hat_a[cur_arm] = np.dot(inv(self.A_a[cur_arm]), self.b_a[cur_arm].transpose())

    def get_payoffs(self, context):
        if len(context) != self.d:
            raise ValueError('Context does not have the same dimentions as given in the constructor')

        payoffs = np.zeros(self.n)

        x = np.array(context).reshape((-1,1)) # 1차원 배열을 열벡터로 변환 (n,) -> (n,1)
        x_t = x.transpose()

        for i in range(self.n):
            firstProduct = np.dot(self.theta_hat_a[i].transpose(), x)
            secondProduct = np.dot(np.dot(x_t, inv(self.A_a[i])), x)

            secondElementSqTimesAlpha = self.alpha * np.sqrt(np.abs(secondProduct))
            payoffs[i] = firstProduct + secondElementSqTimesAlpha

        return payoffs

    def choose_arm(self, context):
        payoffs = self.get_payoffs(context)

        # np.argmax(payoffs)
        max_payoff = float('-inf')
        viable_arms = []

        for i in range(self.n):
            if payoffs[i] > max_payoff:
                viable_arms = [i]
                max_payoff = payoffs[i]
            elif payoffs[i] == max_payoff:
                viable_arms.append(i)

        if not viable_arms:
            raise ValueError('No viable arm!')
        elif len(viable_arms) == 1:
            return viable_arms[0]
        else:
            return np.random.choice(viable_arms)


class EGreedyAgent:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(self.n_arms)
        self.action_counts = np.zeros(self.n_arms)

    def choose_arm(self):
        if np.random.rand() < self.epsilon:
            selected_arm = np.random.choice(self.n_arms)
        else:
            selected_arm = np.argmax(self.q_values)

        return selected_arm

    def update_q_values(self, arm, reward):
        self.action_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.action_counts[arm]



if __name__ == '__main__':

    # Create an instance and Toy example
    linucb = LinUCB(d=5, n=3, alpha=0.1)

    user_contexts = [
        np.array([1.2, 0.8, 0.5, 0.3, 0.2]),
        np.array([0.9, 1.0, 0.3, 0.5, 0.8]),
        np.array([0.5, 0.3, 0.9, 1.0, 0.7])
    ]
    arms_selected = [linucb.choose_arm(context) for context in user_contexts]
    rewards = [2.3, 1.8, 2.5]

    linucb.receive_rewards(user_contexts, arms_selected, rewards)

    new_user_context = np.array([0.2, 1.0, 0.4, 0.9, 0.9])

    payoffs = linucb.get_payoffs(new_user_context)
    print('Payoffs for each arm', payoffs)

    chosen_arm = linucb.choose_arm(new_user_context)
    print('Chosen Arm', chosen_arm)


    # Simulation of product recommendation with LinUCB agent
    def simulate_linucb_agent(agent, contexts, rounds):
        cumulative_rewards = []
        
        for _ in range(rounds):
            user_context = np.random.rand(5)
            linucb_arm = agent.choose_arm(user_context)

            # get reward
            reward = np.random.normal(loc=3.0, scale=1.0)

            #update the theta_hat
            agent.receive_reward(user_context, linucb_arm, reward)

            cumulative_rewards.append(cumulative_rewards[-1] + reward if cumulative_rewards else reward)

        return cumulative_rewards

    def simulate_egreedy_agent(agent, rounds):
        cumulative_rewards = []

        for _ in range(rounds):
            e_greedy_arm = agent.choose_arm()

            reward = np.random.normal(loc=3.0, scale=1.0)

            # update the q_values
            agent.update_q_values(e_greedy_arm, reward)

            cumulative_rewards.append(cumulative_rewards[-1] + reward if cumulative_rewards else reward)

        return cumulative_rewards

    linucb_agent = LinUCB(d=5, n=5, alpha=0.1)
    e_greedy_agent = EGreedyAgent(n_arms=5, epsilon=0.3)

    rounds = 1000
    np.random.seed(42)

    linucb_cumulative_rewards = simulate_linucb_agent(linucb_agent, user_contexts, rounds)
    e_greedy_cumulative_rewards = simulate_egreedy_agent(e_greedy_agent, rounds)

    # Visualize
    plt.plot(linucb_cumulative_rewards, label='LinUCB Agent')
    plt.plot(e_greedy_cumulative_rewards, label='E-Greedy Agent')
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Rewards')
    plt.legend()
    plt.title('Comparision of LinUCB and E-Greedy Agent')

