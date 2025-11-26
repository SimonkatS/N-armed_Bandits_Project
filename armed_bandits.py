import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE ENVIRONMENT (UPDATED) ---
class BanditEnvironment:
    """
    Simulates a set of M bandit machines, each with N arms.
    
    Can be initialized with custom reward ranges for each machine.
    """
    def __init__(self, m, n, machine_ranges=None):
        self.m = m # Number of machines
        self.n = n # Number of arms per machine
        
        # --- NEW: Create true reward values based on ranges ---
        if machine_ranges is None:
            # Default behavior: All arm means are from N(0, 1)
            print("Initializing environment with default random rewards N(0, 1)")
            self.true_rewards = np.random.randn(m, n)
        else:
            # Custom behavior: Use provided ranges
            if len(machine_ranges) != m:
                raise ValueError(f"The 'machine_ranges' list must have {m} elements (one for each machine).")
            
            print("Initializing environment with custom reward ranges.")
            # We need to create an (m, n) array to store the rewards
            self.true_rewards = np.zeros((m, n))
            
            # Loop through each machine and assign rewards
            for i in range(m):
                min_r, max_r = machine_ranges[i]
                if min_r > max_r:
                    raise ValueError(f"Invalid range for machine {i}: min ({min_r}) > max ({max_r})")
                
                # For machine 'i', create 'n' true arm rewards,
                # each uniformly sampled from [min_r, max_r]
                self.true_rewards[i, :] = np.random.uniform(low=min_r, high=max_r, size=n)
                
    def pull_lever(self, machine_idx, arm_idx):
        """
        Pull a specific lever on a specific machine.
        
        Returns a stochastic reward based on the true mean of that lever.
        We simulate this by adding random noise (mean 0, std dev 1)
        to the true mean reward.
        """
        # Get the true mean reward we set during initialization
        true_mean = self.true_rewards[machine_idx, arm_idx]
        
        # Add noise to make it stochastic.
        # This means the *average* reward is true_mean, but any
        # single pull will be a bit different.
        reward = true_mean + np.random.randn() 
        return reward

# --- 2. THE E-GREEDY AGENT ---
class EpsilonGreedyAgent:
    """
    Solves the M x N bandit problem using the Epsilon-Greedy strategy.
    """
    def __init__(self, m, n, epsilon):
        self.m = m
        self.n = n
        self.epsilon = epsilon
        
        # Q-table: Estimated value of each (machine, arm) pair
        # Initialize all estimates to 0
        self.q_table = np.zeros((m, n))
        
        # Action counts: Number of times we've pulled each (machine, arm)
        self.action_counts = np.zeros((m, n))

    def choose_action(self):
        """
        Chooses an action (a machine and an arm) using e-greedy logic.
        """
        if np.random.rand() < self.epsilon:
            # --- EXPLORE ---
            # Choose a random machine and a random arm
            machine = np.random.randint(self.m)
            arm = np.random.randint(self.n)
        else:
            # --- EXPLOIT ---
            # Find the (machine, arm) pair with the highest Q-value
            # np.argmax finds the *flat* index of the max value
            # np.unravel_index converts that flat index back to (row, col)
            flat_index = np.argmax(self.q_table)
            machine, arm = np.unravel_index(flat_index, (self.m, self.n))
            
        return machine, arm

    def update_q_value(self, machine, arm, reward):
        """
        Updates the Q-table using the incremental average formula.
        """
        # Increment the count for this (machine, arm) pair
        self.action_counts[machine, arm] += 1
        k = self.action_counts[machine, arm]
        
        # Get the old Q-value
        old_q = self.q_table[machine, arm]
        
        # Incremental average formula:
        # Q_new = Q_old + (1/k) * (Reward - Q_old)
        new_q = old_q + (1/k) * (reward - old_q)
        
        self.q_table[machine, arm] = new_q

# --- 3. THE SOFTMAX AGENT ---
class SoftmaxAgent:
    """
    Solves the M x N bandit problem using the Softmax (Boltzmann) strategy.
    """
    def __init__(self, m, n, tau):
        self.m = m
        self.n = n
        self.tau = tau # Temperature parameter
        self.q_table = np.zeros((m, n))
        self.action_counts = np.zeros((m, n))

    def choose_action(self):
        """
        Chooses an action by sampling from a probability distribution
        created from the Q-values.
        """
        # Calculate probabilities for all M*N actions
        # P(a) = exp(Q(a) / tau) / sum(exp(Q(all) / tau))
        
        # 1. Get Q-values and divide by temperature
        q_over_tau = self.q_table / self.tau
        
        # 2. To prevent numerical overflow, subtract the max value
        #    This is a standard trick: exp(x - max(x))
        q_over_tau_stable = q_over_tau - np.max(q_over_tau)
        
        # 3. Calculate exponentiated values
        exp_values = np.exp(q_over_tau_stable)
        
        # 4. Calculate sum
        total_sum = np.sum(exp_values)
        
        # 5. Calculate probabilities
        probs = exp_values / total_sum
        
        # Now, choose an action based on these probabilities
        # We flatten the 2D probability array to 1D
        flat_probs = probs.flatten()
        
        # np.random.choice picks one index from 0 to (M*N - 1)
        # based on the probabilities in flat_probs
        total_actions = self.m * self.n
        flat_index = np.random.choice(total_actions, p=flat_probs)
        
        # Convert the flat index back to 2D (machine, arm)
        machine, arm = np.unravel_index(flat_index, (self.m, self.n))
        
        return machine, arm

    def update_q_value(self, machine, arm, reward):
        """
        Updates the Q-table (same logic as Epsilon-Greedy).
        """
        self.action_counts[machine, arm] += 1
        k = self.action_counts[machine, arm]
        old_q = self.q_table[machine, arm]
        new_q = old_q + (1/k) * (reward - old_q)
        self.q_table[machine, arm] = new_q


# --- 4. PUTTING IT ALL TOGETHER (SIMULATION) ---

# --- Hyperparameters ---
M_MACHINES = 5      # Number of machines (m)
N_ARMS = 10         # Number of arms (n)
TOTAL_STEPS = 1000  # Number of "energies" or plays (P)
EPSILON = 0.1       # Epsilon for e-greedy
TAU = 0.1           # Temperature for softmax

# --- Define custom reward ranges for each machine [ (min_mean, max_mean) ] ---
# This is a list with M_MACHINES elements.
custom_ranges = [
    (10, 20),   
    (5, 15),    
    (50, 70),   
    (25, 35),   
    (-10, 0)    
]

print(f"Creating a {M_MACHINES}x{N_ARMS} bandit problem.")
# --- Initialize Environment and Agents ---

env = BanditEnvironment(M_MACHINES, N_ARMS)                                # ---INTERCHANGE THESE 2 LINES IF YOU DONT/DO WANT CUSTOM RANGES---
# env = BanditEnvironment(M_MACHINES, N_ARMS, machine_ranges=custom_ranges)

print(f"\nTrue reward landscape (mean values):")
print(np.round(env.true_rewards, 2))

e_greedy_agent = EpsilonGreedyAgent(M_MACHINES, N_ARMS, EPSILON)
softmax_agent = SoftmaxAgent(M_MACHINES, N_ARMS, TAU)

# --- Store rewards over time ---
e_greedy_rewards = []
softmax_rewards = []

# --- Run Epsilon-Greedy Simulation ---
print("Running Epsilon-Greedy Agent...")
for step in range(TOTAL_STEPS):
    # 1. Agent chooses action
    machine, arm = e_greedy_agent.choose_action()
    
    # 2. Environment gives reward
    reward = env.pull_lever(machine, arm)
    
    # 3. Agent updates its knowledge
    e_greedy_agent.update_q_value(machine, arm, reward)
    
    # 4. Store reward
    e_greedy_rewards.append(reward)


# NOTE: We re-use the *same* environment so the agents face the exact same problem, making it a fair comparison.


# --- Run Softmax Simulation ---
print("Running Softmax Agent...")
for step in range(TOTAL_STEPS):
    # 1. Agent chooses action
    machine, arm = softmax_agent.choose_action()
    
    # 2. Environment gives reward
    reward = env.pull_lever(machine, arm)
    
    # 3. Agent updates its knowledge
    softmax_agent.update_q_value(machine, arm, reward)
    
    # 4. Store reward
    softmax_rewards.append(reward)

# --- Results ---
print("\n--- Simulation Complete ---")

# Find the true best action and its reward
true_best_flat_index = np.argmax(env.true_rewards)
true_best_machine, true_best_arm = np.unravel_index(true_best_flat_index, (M_MACHINES, N_ARMS))
true_best_reward = env.true_rewards[true_best_machine, true_best_arm]


print(f"True best action: Machine {true_best_machine}, Arm {true_best_arm} (True Mean Reward: {true_best_reward:.4f})")

# E-Greedy results
total_e_greedy_reward = np.sum(e_greedy_rewards)
print(f"\nE-Greedy Total Reward: {total_e_greedy_reward:.4f}")
# Find what e-greedy *thinks* is the best action
e_greedy_best_flat_index = np.argmax(e_greedy_agent.q_table)
e_greedy_machine, e_greedy_arm = np.unravel_index(e_greedy_best_flat_index, (M_MACHINES, N_ARMS))
print(f"E-Greedy found: Machine {e_greedy_machine}, Arm {e_greedy_arm} (Estimated Reward: {e_greedy_agent.q_table[e_greedy_machine, e_greedy_arm]:.4f})")


# Softmax results
total_softmax_reward = np.sum(softmax_rewards)
print(f"\nSoftmax Total Reward: {total_softmax_reward:.4f}")
# Find what softmax *thinks* is the best action
softmax_best_flat_index = np.argmax(softmax_agent.q_table)
softmax_machine, softmax_arm = np.unravel_index(softmax_best_flat_index, (M_MACHINES, N_ARMS))
print(f"Softmax found: Machine {softmax_machine}, Arm {softmax_arm} (Estimated Reward: {softmax_agent.q_table[softmax_machine, softmax_arm]:.4f})")


# --- Plotting ---
# Calculate moving average to see learning trend
def moving_average(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(moving_average(e_greedy_rewards), label=f'Epsilon-Greedy (e={EPSILON})')
plt.plot(moving_average(softmax_rewards), label=f'Softmax (tau={TAU})')
plt.axhline(y=true_best_reward, color='r', linestyle='--', label=f'True Optimal Reward ({true_best_reward:.2f})')
plt.title('Agent Performance (Moving Average of Rewards)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()