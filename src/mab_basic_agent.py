import numpy as np
import matplotlib.pyplot as plt

class ProductRecommendationBandit:

    def __init__(self, num_products, recommendation_probabilities, product_rewards):
        if len(recommendation_probabilities) != len(product_rewards):
            raise ValueError('Size of recommendation probabilities does not match the size of product rewards')
        
        self.num_products = num_products
        self.recommendation_probabilities = recommendation_probabilities
        self.product_rewards = product_rewards

    def get_recommendation(self, product):
        if not (0 <= product <= self.num_products):
            raise ValueError(f'Product index must be between 0 and {self.num_products - 1}')

        return np.random.choice([self.product_rewards[product], 0.0],
                                 p = [self.recommendation_probabilities[product], 1 - self.recommendation_probabilities[product]])


class stochasticProductRecommendationBandit:
    def __init__(self, num_products, seed = 42):
        self.num_products = num_products
        self.seed = seed

        # 랜덤 시드 고정
        np.random.seed(self.seed)

        # 정규 분포 랜덤 확률 생성        
        self.recommendation_probabilities = np.clip(np.random.normal(loc=0.5, scale=0.2, size=self.num_products), 0, 1)

        # 랜덤 보상 생성
        self.product_rewards = np.clip(np.random.normal(loc=0.5, scale=0.2, size=self.num_products), 0, 1)

    def get_recommendation(self, product):
        if not (0 <= product <= self.num_products):
            raise ValueError(f'Product index must be between 0 and {self.num_products - 1}')
        
        return np.random.choice([self.product_rewards[product], 0.0],
                                 p = [self.recommendation_probabilities[product], 1 - self.recommendation_probabilities[product]])


class PolicyAgents:

    def __init__(self, enviroment, agent_type, epsilon=0.01, tau=1, n_th_iteration=None, C=2):
        self.enviroment = enviroment
        self.agent_type = agent_type
        self.q_values = np.zeros(self.enviroment.num_products)
        self.rewards = []
        self.selected_arm_list = []
        self.cum_rewards = [0.0]
        self.total_reward = 0
        self.arm_counts = np.zeros(self.enviroment.num_products)

        # epsilon-greedy 에이전트를 위한 epsilon 값
        self.epsilon = epsilon

        # Softmax 에이전트를 위한 tau 값
        self.action_prob = np.zeros(self.enviroment.num_products)
        self.tau = tau

        # UCB 에이전트를 위한 파라미터 값
        self.n_th_iteration = n_th_iteration
        self.C = C

        # Regret 계산을 위한 변수
        self.regret = []
        self.greedy_reward_list = []
        self.total_reward_list = []
        self.best_total_reward = 0
        self.total_regret = 0
        self.total_reward_episode = 0

    def choose_action(self):
        if self.agent_type == 'random':
            selected_arm = np.random.randint(self.enviroment.num_products)
        elif self.agent_type == 'greedy':
            selected_arm = np.argmax(self.q_values)
        elif self.agent_type == 'egreedy':
            if np.random.random() < self.epsilon:
                selected_arm = np.random.randint(self.enviroment.num_products)
            else:
                selected_arm = np.argmax(self.q_values)
        elif self.agent_type == 'softmax':
            self.action_prob = np.exp(self.q_values / self.tau) / np.sum(np.exp(self.q_values / self.tau))
            selected_arm = np.random.choice(self.enviroment.num_products, p=self.action_prob)
        elif self.agent_type == 'ucb':
            if self.n_th_iteration < len(self.q_values):
                selected_arm = self.n_th_iteration
            else:
                U = self.C * np.sqrt(np.log(self.n_th_iteration) / self.arm_counts)
                selected_arm = np.argmax(self.q_values + U)
        else:
            raise ValueError(f'Invalid agent type: {self.agent_type}')

        return selected_arm

    def take_action(self):
        chosen_product = self.choose_action()
        self.selected_arm_list.append(chosen_product)
        reward = self.enviroment.get_recommendation(chosen_product)

        self.rewards.append(reward)
        self.arm_counts[chosen_product] += 1
        self.total_reward += reward

        # update Q-values for chosen product
        self.q_values[chosen_product] = self.q_values[chosen_product] + (1 / self.arm_counts[chosen_product]) * (reward - self.q_values[chosen_product])

        self.cum_rewards.append(sum(self.rewards) / len(self.rewards))

        # Regret 계산
        self.total_reward_episode += reward
        self.total_reward_list.append(self.total_reward_episode)
        best_possible_product = np.argmax(self.q_values)
        best_possible_reward = self.enviroment.get_recommendation(best_possible_product)
        self.best_total_reward += best_possible_reward
        self.greedy_reward_list.append(self.best_total_reward)
        regret = abs(best_possible_reward - reward)
        self.total_regret += regret
        self.regret.append(self.total_regret)

        return chosen_product, reward

    def get_total_reward(self):
        return self.total_reward

    def get_q_values(self):
        return self.q_values

    def get_reward_distribution(self):
        reward_dict = {}

        for arm, reward in zip(self.selected_arm_list, self.rewards):
            if arm not in reward_dict:
                reward_dict[arm] = []

            reward_dict[arm].append(reward)
        return reward_dict

    def get_regret(self):
        return self.regret

    def get_reward_list(self):
        return self.total_reward_list

    def get_greedy_reward_list(self):
        return self.greedy_reward_list



# Plot Function
def plot_cumulative_rewards_and_counts(cum_rewards, arm_counts, reward_dict):

    # 베이스 라인 그리기
    baseline = [1.0] * len(cum_rewards)
    
    plt.figure(figsize=(18, 6))

    # 1st plot: 누적 보상 그래프
    plt.subplot(1, 3, 1)
    plt.plot(cum_rewards, label='Cumulative Rewards')
    plt.plot(baseline, label='Baseline (1.0)', linestyle='--', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Rewards')
    plt.legend()

    # 2nd plot: 각 arm이 선택된 횟수 파이 차트
    plt.subplot(1, 3, 2)
    plt.pie(arm_counts, labels = range(len(arm_counts)), autopct='%1.1f%%', startangle=140)
    plt.title('Number of time each product is selected')
    plt.axis('equal')

    # 3rd plot: 보상 분포 바이올린 차트
    plt.subplot(1, 3, 3)
    rewards_data = [reward_dict[key] for key in sorted(reward_dict.keys())]
    plt.violinplot(rewards_data, showmedians=True, showextrema=False)
    plt.xticks(range(1, len(reward_dict) + 1), sorted(reward_dict.keys()))
    plt.title('Reward distribution for each product')
    plt.xlabel('Product')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.show()

# Epsilon 파라미터 튜닝
def run_agent_with_epsilon(selected_epsilon):
    egreedy_agent = PolicyAgents(env, agent_type = 'egreedy', epsilon=selected_epsilon)

    total_rewards = []

    num_steps = 1000
    for _ in range(num_steps):
        chosen_product, reward = egreedy_agent.take_action()
    
    egreedy_total_reward = egreedy_agent.get_total_reward()
    total_rewards.append(egreedy_total_reward)

    return total_rewards

# Plot Epsilon 파라미터 튜닝 결과
def plot_results(result_dicts):
    labels = list(result_dicts.keys())
    total_rewards = list(result_dicts.values())
    
    x = np.arange(len(labels))

    width = 0.35

    fig, ax = plt.subplots()
    total_rewards = np.array(total_rewards)

    for i in range(len(labels)):
        ax.bar(x[i], total_rewards[i], width, label = f'{labels[i]}')

    ax.set_ylabel('Total Rewards')
    ax.set_title('Total Reward by Epsilon Value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    plt.show()

# Plot Regret
def regret_curve(x_values, y_lists, labels, title, xlabel, ylabel):

    fig, axes = plt.subplots(1,2, figsize=(18,6))

    for y_list, label in zip(y_lists, labels):
        axes[0].plot(x_values, y_list, label=label)
    
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(True)

    regret_label = labels[1]
    axes[1].plot(x_values, y_lists[1], label=regret_label)
    axes[1].set_title(f'{title}')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()
    axes[1].grid(True)

    plt.show()



if __name__ == '__main__':
    # # 랜덤 환경 인스턴스 생성
    # recom_probabilities = [0.52, 0.08, 0.87, 0.56]
    # env = ProductRecommendationBandit(num_products=4, recommendation_probabilities=recom_probabilities, product_rewards=[1.0, 1.0, 1.0, 1.0])

    # 확률적 환경 인스턴스 생성
    num_products = 4
    env = stochasticProductRecommendationBandit(num_products=num_products)

    print("Recommendation Probabilities: ", env.recommendation_probabilities)
    print("Product Rewards: ", env.product_rewards)

    # # n번째 상품의 10번 추천 결과
    # results = [env.get_recommendation(2) for _ in range(10)]
    # print(results)


    # # Epsilon 파라미터 튜닝
    # epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]

    # # Epsilon 값 별로 에이전트 실행
    # results_dict = {}
    # for epsilon_value in epsilon_values:
    #     total_rewards = run_agent_with_epsilon(selected_epsilon=epsilon_value)
    #     results_dict[f'Epsilon: {epsilon_value}'] = total_rewards
    
    # plot_results(results_dict)


    # # 랜덤 에이전트
    # random_agent = PolicyAgents(env, 'random')

    # # 랜덤 에이전트 1000번 추천 실행
    # num_steps = 1000
    # for _ in range(num_steps):
    #     chosen_product, reward = random_agent.take_action()

    # random_total_reward = random_agent.get_total_reward()
    # random_q_values = random_agent.get_q_values()

    # print(f'랜덤 에이전트의 총 보상: {random_total_reward}')
    # print(f'랜덤 에이전트의 Q-values: {random_q_values}')
    # print(f'실제 추천 확률: {env.recommendation_probabilities}')

    # random_cum_rewards = random_agent.cum_rewards
    # random_arm_counts = random_agent.arm_counts
    # random_reward_distribution = random_agent.get_reward_distribution()

    # plot_cumulative_rewards_and_counts(random_cum_rewards, random_arm_counts, random_reward_distribution)

    # # 그리디 에이전트
    # greedy_agent = PolicyAgents(env, 'greedy')

    # # 그리디 에이전트 1000번 추천 실행
    # num_steps = 1000
    # for _ in range(num_steps):
    #     chosen_product, reward = greedy_agent.take_action()

    # greedy_total_reward = greedy_agent.get_total_reward()
    # greedy_q_values = greedy_agent.get_q_values()

    # print(f'그리디 에이전트의 총 보상: {greedy_total_reward}')
    # print(f'그리디 에이전트의 Q-values: {greedy_q_values}')
    # print(f'실제 추천 확률: {env.recommendation_probabilities}')

    # greedy_cum_rewards = greedy_agent.cum_rewards
    # greedy_arm_counts = greedy_agent.arm_counts
    # greedy_reward_distribution = greedy_agent.get_reward_distribution()

    # plot_cumulative_rewards_and_counts(greedy_cum_rewards, greedy_arm_counts, greedy_reward_distribution)

    # # E-그리디 에이전트
    # egreedy_agent = PolicyAgents(env, 'egreedy')

    # # E-그리디 에이전트 1000번 추천 실행
    # num_steps = 1000
    # for _ in range(num_steps):
    #     chosen_product, reward = egreedy_agent.take_action()

    # egreedy_total_reward = egreedy_agent.get_total_reward()
    # egreedy_q_values = egreedy_agent.get_q_values()

    # print(f'E-그리디 에이전트의 총 보상: {egreedy_total_reward}')
    # print(f'E-그리디 에이전트의 Q-values: {egreedy_q_values}')
    # print(f'실제 추천 확률: {env.recommendation_probabilities}')

    # egreedy_cum_rewards = egreedy_agent.cum_rewards
    # egreedy_arm_counts = egreedy_agent.arm_counts
    # egreedy_reward_distribution = egreedy_agent.get_reward_distribution()

    # plot_cumulative_rewards_and_counts(egreedy_cum_rewards, egreedy_arm_counts, egreedy_reward_distribution)

    # # Softmax 에이전트
    # softmax_agent = PolicyAgents(env, 'softmax', tau=0.5)

    # # Softmax 에이전트 1000번 추천 실행
    # num_steps = 1000
    # for _ in range(num_steps):
    #     chosen_product, reward = softmax_agent.take_action()

    # softmax_total_reward = softmax_agent.get_total_reward()
    # softmax_q_values = softmax_agent.get_q_values()

    # print(f'Softmax 에이전트의 총 보상: {softmax_total_reward}')
    # print(f'Softmax 에이전트의 Q-values: {softmax_q_values}')
    # print(f'실제 추천 확률: {env.recommendation_probabilities}')

    # softmax_cum_rewards = softmax_agent.cum_rewards
    # softmax_arm_counts = softmax_agent.arm_counts
    # softmax_reward_distribution = softmax_agent.get_reward_distribution()

    # plot_cumulative_rewards_and_counts(softmax_cum_rewards, softmax_arm_counts, softmax_reward_distribution)

    # # UCB 에이전트
    # ucb_agent = PolicyAgents(env, 'ucb', C=1)

    # # UCB 에이전트 1000번 추천 실행
    # num_steps = 1000
    # for i in range(num_steps):
    #     ucb_agent.n_th_iteration = i
    #     chosen_product, reward = ucb_agent.take_action()

    # ucb_total_reward = ucb_agent.get_total_reward()
    # ucb_q_values = ucb_agent.get_q_values()

    # print(f'UCB 에이전트의 총 보상: {ucb_total_reward}')
    # print(f'UCB 에이전트의 Q-values: {ucb_q_values}')
    # print(f'실제 추천 확률: {env.recommendation_probabilities}')

    # ucb_cum_rewards = ucb_agent.cum_rewards
    # ucb_arm_counts = ucb_agent.arm_counts
    # ucb_reward_distribution = ucb_agent.get_reward_distribution()

    # plot_cumulative_rewards_and_counts(ucb_cum_rewards, ucb_arm_counts, ucb_reward_distribution)


    # # 모든 에이전트의 누적 보상을 하나의 차트로 비교
    # ln_width = 2
    # x_values = list(range(len(random_cum_rewards)))

    # plt.figure(figsize=(18, 6))

    # plt.plot(x_values, random_cum_rewards, label='Random', linestyle='-', linewidth=ln_width)
    # plt.plot(x_values, greedy_cum_rewards, label='Greedy', linestyle='--', linewidth=ln_width)
    # plt.plot(x_values, egreedy_cum_rewards, label='E-Greedy', linestyle='-.', linewidth=ln_width)
    # plt.plot(x_values, softmax_cum_rewards, label='Softmax', linestyle=':', linewidth=ln_width)
    # plt.plot(x_values, ucb_cum_rewards, label='UCB', linestyle='-', linewidth=ln_width)

    # plt.xlabel('Episodes')
    # plt.ylabel('Cumulative Rewards')
    # plt.title('Comparison of Cumulative Rewards of Different Agents')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    
    # E-그리디 에이전트 Regret 계산
    egreedy_agent = PolicyAgents(env, 'egreedy', epsilon=0.01)

    # E-그리디 에이전트 1000번 추천 실행
    num_steps = 1000
    for _ in range(num_steps):
        chosen_product, reward = egreedy_agent.take_action()

    reward_list = egreedy_agent.get_reward_list()
    regret_list = egreedy_agent.get_regret()
    egreedy_reward_list = egreedy_agent.get_greedy_reward_list()

    steps = range(len(reward_list))
    y_lists = [reward_list, regret_list, egreedy_reward_list]
    labels = ['Reward', 'Regret', 'E-Greedy Reward']
    
    regret_curve(steps, y_lists, labels, 'Regret Curves Stochastic', 'Steps', 'Values')