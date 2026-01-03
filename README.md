# 🏓 Pong Reinforcement Learning Agent

An **advanced Deep Reinforcement Learning project** where an agent learns to play **Pong** using modern **Deep Q-Learning** techniques.
The agent is trained from scratch in a **custom Gymnasium environment**, gradually improving through curriculum learning and sophisticated replay strategies.

---

## ✨ Key Features

* **Custom Pong Environment** (Gymnasium + Pygame)
* **Dueling Deep Q-Network (Dueling DQN)**
* **Double DQN** for reduced overestimation bias
* **Prioritized Experience Replay (PER)**
* **Curriculum Learning** with increasing opponent difficulty
* **Reward Shaping** for faster convergence
* **Soft & Hard Target Network Updates**
* **Detailed Training & Testing Analytics**
* **Benchmarking Against Multiple Opponent Skill Levels**

---

## 📂 Project Structure

```text
.
├── dqn_agent.py        # Dueling DQN agent + Prioritized Replay Buffer
├── pong_env.py         # Custom Pong Gymnasium environment
├── train.py            # Advanced training pipeline with curriculum learning
├── test.py             # Evaluation, benchmarking, and visualization
├── checkpoints/        # Saved model checkpoints (auto-created)
├── *.pth               # Trained model files
└── README.md           # Project documentation
```

---

## 🧠 Reinforcement Learning Architecture

### Agent

* **Neural Network**: Dueling DQN

  * Shared feature extractor
  * Separate **Value** and **Advantage** streams
* **Loss Function**: Huber Loss
* **Optimizer**: Adam with weight decay
* **Exploration**: ε-greedy with exponential decay

### Replay Strategy

* **Prioritized Experience Replay**
* Importance Sampling correction with annealing β

---

## 🎮 Environment Details

* **State Space (8 dimensions)**:

  * Ball position (x, y)
  * Ball velocity (x, y)
  * Agent paddle position
  * Opponent paddle position
  * Distance from ball to paddle
  * Ball direction indicator

* **Action Space (3 actions)**:

  * `0` → Stay
  * `1` → Move Up
  * `2` → Move Down

* **Reward Shaping Includes**:

  * Positive reward for paddle-ball alignment
  * Increasing rewards for longer rallies
  * High reward for scoring
  * Penalty for missing the ball
  * Small penalty for inaction

---

## 🏋️ Training the Agent

Start training from scratch using curriculum learning:

```bash
python train.py
```

What happens during training:

* Opponent skill increases over 4 stages (Beginner → Expert)
* Models are checkpointed periodically
* Best model is automatically saved
* Training statistics and plots are generated

Final trained model:

```text
pong_agent_advanced_final.pth
```

📌 Training logic reference: 

---

## 🧪 Testing the Agent

### Test with Rendering

```bash
python test.py pong_agent_advanced_final.pth
```

### Custom Opponent Skill & Episodes

```bash
python test.py pong_agent_advanced_final.pth 0.85 50
```

### Full Benchmark

```bash
python test.py benchmark
```

Testing outputs:

* Win rate
* Average score
* Rally length
* Reward statistics
* Performance plots

📌 Testing logic reference: 

---

## 📊 Visual Outputs

During training and testing, the following plots are generated:

* Reward vs Episodes
* Win Rate vs Time
* Rally Length Distribution
* Loss Curves
* Benchmark Performance Graphs

Saved as:

```text
advanced_training_progress.png
test_results.png
benchmark_results.png
```

---

## 💾 Saving & Loading Models

### Save (automatic during training)

```python
agent.save("model.pth")
```

### Load for testing or continued training

```python
agent.load("model.pth")
```

📌 Agent implementation reference: 

---

## 🕹️ Custom Pong Environment

* Built using **Gymnasium API**
* Rendered using **Pygame**
* Optimized `fast_mode` for high-speed training
* Human-playable and visually debuggable

📌 Environment implementation reference: 
