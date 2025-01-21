import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import cv2
import datetime

print("Code begins...")

# Paths for saving model and videos
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop/aigame')
videos_folder = os.path.join(desktop_path, 'ai_videos')
model_folder = os.path.join(desktop_path, 'ai_model')
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# Load CartPole dataset
dataset_path = "C:/Users/Perdorues/Desktop/aigame/cartpole_dataset.csv"
cartpole_data = pd.read_csv(dataset_path)

# Define the DQN neural network model
class DQNNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQNNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNNetwork(action_size)
        self.target_model = DQNNetwork(action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.amax(self.target_model(next_state)[0]) * (1 - done)
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            target = np.reshape(target, (1,))
            loss = tf.keras.losses.mean_squared_error(target, q_values[0][action])
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# Create CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = os.path.join(videos_folder, 'cartpole_video.avi')
video_recorder = cv2.VideoWriter(video_path, fourcc, 10.0, (600, 400))

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

# Load the latest model automatically
checkpoint_files = [f for f in os.listdir(model_folder) if f.endswith(".index")]
if checkpoint_files:
    latest_model = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(model_folder, x)))
    latest_model_path = os.path.join(model_folder, latest_model.replace(".index", ""))
    print(f"Loading the latest model: {latest_model_path}")
    agent.model.load_weights(latest_model_path)
    agent.target_model.load_weights(latest_model_path)
    print("Model loaded successfully.")
else:
    print("No pre-trained model found, starting from scratch.")

# Train the agent using dataset
print("Training using dataset...")
for idx, row in cartpole_data.iterrows():
    print(f"Training with dataset row {idx + 1}/{len(cartpole_data)}")

    state = np.array([[row['cart_position'], row['cart_velocity'], row['pole_angle'], row['pole_angular_velocity']]])
    action = int(row['action'])
    reward = row['reward']
    next_state = np.array([[row['cart_position'] + 0.01, row['cart_velocity'] - 0.01, row['pole_angle'] + 0.005, row['pole_angular_velocity'] - 0.005]])
    done = int(row['done'])

    agent.train(state, action, reward, next_state, done)

agent.update_target_model()
print("Pre-training from dataset complete.")

# Fine-tune using reinforcement learning
episodes = 7000
reward_history = []

for episode in range(episodes):
    initial_state = env.reset()[0]
    state = np.reshape(initial_state, [1, state_size])

    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print(f"Episode {episode + 1}, Step {step_count}, Action: {action}, Reward: {reward}")

        next_state = np.reshape(next_state, [1, state_size])
        agent.train(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        step_count += 1

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    if total_reward > 195:
        print(f"Early stopping at episode {episode + 1}")
        break

    reward_history.append(total_reward)

video_recorder.release()

# Save the trained model
model_save_path = os.path.join(model_folder, "cartpole_model-" + datetime.datetime.now().strftime("%y%m%d%H%M") + ".tfmodel")
agent.target_model.save_weights(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training performance
plt.plot(reward_history)
plt.title('Training Performance')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
