import os
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import cv2
import datetime
import tkinter as tk
from PIL import Image, ImageTk
import threading

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop/aigame')

videos_folder = os.path.join(desktop_path, 'ai_videos')
model_folder = os.path.join(desktop_path, 'ai_model')
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

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
        self.epsilon = 0.0 
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

root = tk.Tk()
canvas = tk.Canvas(root, width=600, height=400)
canvas.pack()

def update_canvas(img, canvas):
    tk_img = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=tk_img, anchor="nw")
    canvas.image = tk_img  # Keep a reference to avoid garbage collection

def play(model_folder, canvas):
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    latest_model = max([f for f in os.listdir(model_folder) if f.endswith(".tfmodel.index")], 
                       key=lambda x: os.path.getctime(os.path.join(model_folder, x)))
    latest_model_path = os.path.join(model_folder, latest_model.replace(".index", ""))
    agent.model.load_weights(latest_model_path)
    agent.target_model.load_weights(latest_model_path)
    
    agent.epsilon = 0
    episodes = 3
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
            img = Image.fromarray(frame)

            root.after(0, update_canvas, img, canvas)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        reward_history.append(total_reward)

    plt.plot(reward_history)
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

thread = threading.Thread(target=play, args=(model_folder, canvas))
thread.start()

root.mainloop()
