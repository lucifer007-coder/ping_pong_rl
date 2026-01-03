from pong_env import PongEnv
from dqn_agent import DQNAgent
import time
import numpy as np
import matplotlib.pyplot as plt

def test_agent(model_path, episodes=20, opponent_skill=0.8, render=True, detailed=True):
    """Enhanced testing with detailed statistics"""
    render_mode = "human" if render else None
    env = PongEnv(render_mode=render_mode, opponent_skill=opponent_skill, fast_mode=False)
    agent = DQNAgent(state_dim=8, action_dim=3)
    
    try:
        agent.load(model_path)
        print(f"✓ Loaded model from {model_path}")
        print(f"  Training steps: {agent.training_steps}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        return
    
    print(f"\n{'='*70}")
    print(f"TESTING AGAINST OPPONENT (Skill: {opponent_skill})")
    print(f"{'='*70}\n")
    
    total_rewards = []
    total_scores = []
    opponent_scores = []
    rally_lengths = []
    episode_lengths = []
    wins = 0
    losses = 0
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        if detailed:
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        while not done and steps < 2000:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if render:
                time.sleep(0.005)
        
        total_rewards.append(episode_reward)
        total_scores.append(info['score_agent'])
        opponent_scores.append(info['score_opponent'])
        rally_lengths.append(info['rally_length'])
        episode_lengths.append(steps)
        
        if info['score_agent'] > info['score_opponent']:
            wins += 1
            result = "WIN! 🎉"
        elif info['score_agent'] < info['score_opponent']:
            losses += 1
            result = "LOSS 😞"
        else:
            result = "TIE 🤝"
        
        if detailed:
            print(f"Result: {result}")
            print(f"Score: {info['score_agent']} - {info['score_opponent']}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Rally: {info['rally_length']}")
            print(f"Steps: {steps}")
    
    env.close()
    
    # Calculate statistics
    win_rate = wins / episodes
    avg_score = np.mean(total_scores)
    avg_opponent_score = np.mean(opponent_scores)
    avg_reward = np.mean(total_rewards)
    avg_rally = np.mean(rally_lengths)
    avg_steps = np.mean(episode_lengths)
    
    score_diff = avg_score - avg_opponent_score
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Episodes: {episodes}")
    print(f"Opponent Skill: {opponent_skill}")
    print(f"\nResults:")
    print(f"  Wins: {wins} ({win_rate:.1%})")
    print(f"  Losses: {losses} ({losses/episodes:.1%})")
    print(f"  Ties: {episodes - wins - losses}")
    print(f"\nPerformance:")
    print(f"  Avg Agent Score: {avg_score:.2f}")
    print(f"  Avg Opponent Score: {avg_opponent_score:.2f}")
    print(f"  Score Differential: {score_diff:+.2f}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Rally Length: {avg_rally:.1f}")
    print(f"  Avg Episode Length: {avg_steps:.0f} steps")
    print(f"\nConsistency:")
    print(f"  Score StdDev: {np.std(total_scores):.2f}")
    print(f"  Reward StdDev: {np.std(total_rewards):.2f}")
    print(f"{'='*70}")
    
    # Performance rating
    if win_rate >= 0.8:
        rating = "EXCELLENT! 🏆"
    elif win_rate >= 0.6:
        rating = "GOOD! 👍"
    elif win_rate >= 0.4:
        rating = "AVERAGE 😐"
    else:
        rating = "NEEDS IMPROVEMENT 📈"
    
    print(f"\nPerformance Rating: {rating}")
    print(f"{'='*70}\n")
    
    # Plot results
    if episodes >= 10:
        plot_test_results(total_scores, opponent_scores, rally_lengths, total_rewards)
    
    return {
        'win_rate': win_rate,
        'avg_score': avg_score,
        'avg_reward': avg_reward,
        'avg_rally': avg_rally
    }

def plot_test_results(agent_scores, opponent_scores, rallies, rewards):
    """Visualize test results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test Results Analysis', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(agent_scores) + 1)
    
    # Plot 1: Scores comparison
    ax = axes[0, 0]
    ax.plot(episodes, agent_scores, marker='o', label='Agent', linewidth=2, color='blue')
    ax.plot(episodes, opponent_scores, marker='s', label='Opponent', linewidth=2, color='red')
    ax.axhline(y=np.mean(agent_scores), color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(opponent_scores), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Score Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Rally lengths
    ax = axes[0, 1]
    ax.bar(episodes, rallies, color='orange', alpha=0.7)
    ax.axhline(y=np.mean(rallies), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rallies):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rally Length')
    ax.set_title('Rally Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Rewards
    ax = axes[1, 0]
    ax.plot(episodes, rewards, marker='o', linewidth=2, color='green')
    ax.axhline(y=np.mean(rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Win/Loss distribution
    ax = axes[1, 1]
    wins = sum(1 for a, o in zip(agent_scores, opponent_scores) if a > o)
    losses = sum(1 for a, o in zip(agent_scores, opponent_scores) if a < o)
    ties = len(agent_scores) - wins - losses
    
    colors = ['green', 'red', 'gray']
    labels = [f'Wins ({wins})', f'Losses ({losses})', f'Ties ({ties})']
    sizes = [wins, losses, ties]
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Win/Loss Distribution')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Test results plot saved as 'test_results.png'")
    plt.show()

def benchmark_against_multiple_opponents():
    """Test against various opponent skill levels"""
    model_path = "pong_agent_advanced_final.pth"
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BENCHMARK")
    print(f"{'='*70}\n")
    
    skill_levels = [0.3, 0.5, 0.7, 0.85, 0.95]
    results = []
    
    for skill in skill_levels:
        print(f"\nTesting against opponent skill: {skill}")
        print("-" * 70)
        result = test_agent(model_path, episodes=20, opponent_skill=skill, 
                           render=False, detailed=False)
        results.append((skill, result))
        time.sleep(1)
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Skill':<10} {'Win Rate':<12} {'Avg Score':<12} {'Avg Rally':<12}")
    print("-" * 70)
    
    for skill, result in results:
        print(f"{skill:<10.2f} {result['win_rate']:<12.1%} "
              f"{result['avg_score']:<12.2f} {result['avg_rally']:<12.1f}")
    
    print(f"{'='*70}\n")
    
    # Plot benchmark
    skills = [r[0] for r in results]
    win_rates = [r[1]['win_rate'] for r in results]
    avg_scores = [r[1]['avg_score'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(skills, win_rates, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.set_xlabel('Opponent Skill Level')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate vs Opponent Skill')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(skills, avg_scores, marker='s', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Opponent Skill Level')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Average Score vs Opponent Skill')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("✓ Benchmark plot saved as 'benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("PONG RL AGENT - ADVANCED TESTING")
    print("="*70)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            benchmark_against_multiple_opponents()
        else:
            model_path = sys.argv[1]
            opponent_skill = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
            episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            test_agent(model_path, episodes=episodes, opponent_skill=opponent_skill)
    else:
        print("\nUsage:")
        print("  python test.py                              # Test final model")
        print("  python test.py <model_path>                 # Test specific model")
        print("  python test.py <model_path> <skill> <eps>   # Custom test")
        print("  python test.py benchmark                    # Full benchmark")
        print("\nTesting final model...\n")
        test_agent("pong_agent_advanced_final.pth", episodes=20, opponent_skill=0.8)
