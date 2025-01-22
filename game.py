import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import cv2
import datetime

print("Code begins...")

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop/aigame')

videos_folder = os.path.join(desktop_path, 'ai_videos')
model_folder = os.path.join(desktop_path, 'ai_model')
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# Load CartPole dataset
dataset_path = os.path.join(desktop_path, 'cartpole_dataset.csv')
cartpole_data = pd.read_csv(dataset_path)

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

env = gym.make("CartPole-v1", render_mode="rgb_array")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = os.path.join(videos_folder, 'video.avi')
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
    next_state = np.array([[row['cart_position'], row['cart_velocity'], row['pole_angle'], row['pole_angular_velocity']]])
    done = int(row['done'])
    agent.train(state, action, reward, next_state, done)

episodes = 700
reward_history = []

for episode in range(episodes):
    initial_state, _ = env.reset()
    state = np.reshape(initial_state, [1, state_size])
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        frame = env.render()
        if frame is not None:
            frame = cv2.resize(frame, (600, 400))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_recorder.write(frame_bgr)

    if episode % 10 == 0:
        agent.update_target_model()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    reward_history.append(total_reward)

video_recorder.release()
agent.target_model.save_weights(os.path.join(model_folder, "model-" + datetime.datetime.now().strftime("%y%m%d%H%M") + ".tfmodel"))

plt.plot(reward_history)
plt.title('Training Performance')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
