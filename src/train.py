import numpy as np
from pong_env import PongEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import time
import os

def train_advanced(total_episodes=3000, render_every=200, save_every=200):
    """Advanced training with optimizations for faster learning"""
    
    # Initialize agent with advanced features
    agent = DQNAgent(
        state_dim=8,  # Extended state space
        action_dim=3,
        learning_rate=0.0003,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        use_double_dqn=True,
        use_prioritized_replay=True
    )
    
    episode_rewards = []
    episode_scores = []
    rally_lengths = []
    losses = []
    win_rates = []
    
    # Aggressive curriculum - faster progression
    stages = [
        {'episodes': 600, 'opponent_skill': 0.4, 'name': 'Beginner'},
        {'episodes': 600, 'opponent_skill': 0.6, 'name': 'Intermediate'},
        {'episodes': 800, 'opponent_skill': 0.8, 'name': 'Advanced'},
        {'episodes': 1000, 'opponent_skill': 0.95, 'name': 'Expert'},
    ]
    
    episode = 0
    start_time = time.time()
    best_win_rate = 0
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx + 1}/4: {stage['name']} (Opponent Skill: {stage['opponent_skill']})")
        print(f"{'='*70}\n")
        
        env = PongEnv(render_mode=None, opponent_skill=stage['opponent_skill'], fast_mode=True)
        
        stage_wins = 0
        stage_episodes = stage['episodes']
        
        for stage_episode in range(stage_episodes):
            state, info = env.reset()
            episode_reward = 0
            steps = 0
            episode_loss = 0
            loss_count = 0
            
            # Render occasionally
            if episode % render_every == 0 and episode > 0:
                env.render_mode = "human"
            else:
                env.render_mode = None
            
            done = False
            max_steps = 2000
            
            while not done and steps < max_steps:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, terminated)
                
                # Train more frequently for faster learning
                if steps % 2 == 0:  # Every 2 steps instead of 4
                    loss = agent.train()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1
                
                episode_reward += reward
                steps += 1
                state = next_state
            
            # Soft update target network every episode for stability
            agent.soft_update_target_network(tau=0.005)
            
            # Hard update every 10 episodes
            if episode % 10 == 0:
                agent.update_target_network()
            
            agent.decay_epsilon()
            
            # Track wins
            if info['score_agent'] > info['score_opponent']:
                stage_wins += 1
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_scores.append(info['score_agent'])
            rally_lengths.append(info['rally_length'])
            avg_loss = episode_loss / max(loss_count, 1)
            losses.append(avg_loss)
            
            # Calculate rolling win rate
            if episode >= 50:
                recent_scores = episode_scores[-50:]
                recent_wins = sum(1 for s in recent_scores if s > 0)
                win_rate = recent_wins / 50
                win_rates.append(win_rate)
            else:
                win_rates.append(0)
            
            # Print progress
            if episode % 25 == 0:
                avg_reward = np.mean(episode_rewards[-25:])
                avg_score = np.mean(episode_scores[-25:])
                avg_rally = np.mean(rally_lengths[-25:])
                avg_loss_25 = np.mean(losses[-25:]) if losses else 0
                current_win_rate = win_rates[-1] if win_rates else 0
                elapsed = time.time() - start_time
                
                print(f"Ep {episode:4d}/{total_episodes} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Score: {avg_score:4.2f} | "
                      f"Rally: {avg_rally:4.1f} | "
                      f"WinRate: {current_win_rate:5.1%} | "
                      f"Loss: {avg_loss_25:6.4f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Time: {elapsed/60:.1f}m")
            
            episode += 1
            
            # Save checkpoints
            if episode % save_every == 0 and episode > 0:
                agent.save(f"checkpoints/pong_advanced_ep{episode}.pth")
                
                # Save best model
                if win_rates[-1] > best_win_rate:
                    best_win_rate = win_rates[-1]
                    agent.save("checkpoints/pong_best_model.pth")
                    print(f"  → New best model! Win rate: {best_win_rate:.1%}")
        
        env.close()
        
        # Stage summary
        stage_win_rate = stage_wins / stage_episodes
        stage_start = episode - stage_episodes
        stage_rewards = episode_rewards[stage_start:episode]
        stage_scores = episode_scores[stage_start:episode]
        stage_rallies = rally_lengths[stage_start:episode]
        
        print(f"\n{stage['name']} Stage Complete:")
        print(f"  Win Rate: {stage_win_rate*100:.1f}%")
        print(f"  Avg Reward: {np.mean(stage_rewards):.2f}")
        print(f"  Avg Score: {np.mean(stage_scores):.2f}")
        print(f"  Avg Rally: {np.mean(stage_rallies):.1f}")
        print(f"  Current ε: {agent.epsilon:.3f}")
    
    # Final save
    agent.save("pong_agent_advanced_final.pth")
    
    # Plot results
    plot_advanced_training(episode_rewards, episode_scores, rally_lengths, losses, win_rates)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Best Win Rate: {best_win_rate:.1%}")
    print(f"Training Steps: {agent.training_steps}")
    print(f"{'='*70}")
    
    return agent

def plot_advanced_training(rewards, scores, rallies, losses, win_rates):
    """Enhanced visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Advanced Pong RL Training Progress', fontsize=16, fontweight='bold')
    
    episodes = range(len(rewards))
    
    # Plot 1: Rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.2, color='blue', linewidth=0.5)
    if len(rewards) >= 50:
        ma_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax.plot(range(49, len(rewards)), ma_rewards, 
                label='50-ep MA', linewidth=2, color='darkblue')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scores
    ax = axes[0, 1]
    ax.plot(scores, alpha=0.2, color='green', linewidth=0.5)
    if len(scores) >= 50:
        ma_scores = np.convolve(scores, np.ones(50)/50, mode='valid')
        ax.plot(range(49, len(scores)), ma_scores, 
                label='50-ep MA', linewidth=2, color='darkgreen')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Agent Score')
    ax.set_title('Scores Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Win Rate
    ax = axes[0, 2]
    if win_rates:
        ax.plot(win_rates, linewidth=2, color='purple')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% Win Rate')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='70% Win Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (50-ep window)')
    ax.set_title('Win Rate Over Time')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Rally Length
    ax = axes[1, 0]
    ax.plot(rallies, alpha=0.2, color='orange', linewidth=0.5)
    if len(rallies) >= 50:
        ma_rallies = np.convolve(rallies, np.ones(50)/50, mode='valid')
        ax.plot(range(49, len(rallies)), ma_rallies, 
                label='50-ep MA', linewidth=2, color='darkorange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rally Length')
    ax.set_title('Rally Length Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Loss
    ax = axes[1, 1]
    if losses:
        ax.plot(losses, alpha=0.2, color='red', linewidth=0.5)
        if len(losses) >= 50:
            ma_losses = np.convolve(losses, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(losses)), ma_losses, 
                    label='50-ep MA', linewidth=2, color='darkred')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Performance Distribution
    ax = axes[1, 2]
    if len(scores) >= 100:
        # Show distribution of scores in last 500 episodes
        recent_scores = scores[-500:]
        ax.hist(recent_scores, bins=20, color='teal', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(recent_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(recent_scores):.2f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution (Last 500 Episodes)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add stage markers
    stage_boundaries = [600, 1200, 2000]
    stage_names = ['Beg→Int', 'Int→Adv', 'Adv→Exp']
    for ax in axes.flat[:5]:  # Skip histogram
        for boundary, name in zip(stage_boundaries, stage_names):
            if boundary < len(rewards):
                ax.axvline(x=boundary, color='purple', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('advanced_training_progress.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training plot saved as 'advanced_training_progress.png'")
    plt.show()

def continue_training(model_path, additional_episodes=1000, opponent_skill=0.95):
    """Continue training from a checkpoint"""
    print(f"\n{'='*70}")
    print(f"CONTINUING TRAINING FROM CHECKPOINT")
    print(f"{'='*70}\n")
    
    agent = DQNAgent(
        state_dim=8,
        action_dim=3,
        use_double_dqn=True,
        use_prioritized_replay=True
    )
    
    try:
        agent.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Boost exploration slightly
    agent.epsilon = max(0.1, agent.epsilon)
    print(f"  Reset epsilon to: {agent.epsilon:.3f}")
    
    env = PongEnv(render_mode=None, opponent_skill=opponent_skill, fast_mode=True)
    
    episode_rewards = []
    episode_scores = []
    wins = 0
    
    start_time = time.time()
    
    for episode in range(additional_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 2000:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, terminated)
            
            if steps % 2 == 0:
                agent.train()
            
            episode_reward += reward
            steps += 1
            state = next_state
        
        if episode % 5 == 0:
            agent.soft_update_target_network(tau=0.005)
        
        if episode % 10 == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score_agent'])
        
        if info['score_agent'] > info['score_opponent']:
            wins += 1
        
        if episode % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            avg_score = np.mean(episode_scores[-25:])
            win_rate = wins / (episode + 1)
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:4d}/{additional_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Score: {avg_score:4.2f} | "
                  f"WinRate: {win_rate:5.1%} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {elapsed/60:.1f}m")
        
        if episode % 200 == 0 and episode > 0:
            agent.save(f"checkpoints/pong_continued_ep{episode}.pth")
    
    env.close()
    
    agent.save("pong_agent_continued_final.pth")
    
    total_time = time.time() - start_time
    final_win_rate = wins / additional_episodes
    
    print(f"\n{'='*70}")
    print(f"CONTINUED TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Episodes: {additional_episodes}")
    print(f"Win Rate: {final_win_rate:.1%}")
    print(f"Avg Score: {np.mean(episode_scores):.2f}")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    return agent

if __name__ == "__main__":
    import sys
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "continue":
        # Continue training from checkpoint
        model_path = sys.argv[2] if len(sys.argv) > 2 else "pong_agent_advanced_final.pth"
        additional_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        agent = continue_training(model_path, additional_episodes=additional_eps)
    else:
        # Fresh training
        agent = train_advanced(total_episodes=3000, render_every=200, save_every=200)
