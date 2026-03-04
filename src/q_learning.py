import numpy as np
from tqdm import tqdm


def train_q_learning(env, agent, n_episodes=10000, verbose=True):
    rewards_per_episode = []
    wins_per_episode = []
    epsilon_history = []
    
    iterator = tqdm(range(n_episodes), desc="Entrenamiento") if verbose else range(n_episodes)
    
    for episode in iterator:
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Elegir acción usando Epsilon-Greedy
            action = agent.choose_action(state, training=True)
            
            # Ejecutar acción y observar resultado
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Actualizar Q-table
            agent.update_q_value(state, action, reward, next_state)
            
            # Actualizar estado
            state = next_state
            total_reward += reward
        
        # Decay epsilon después de cada episodio
        agent.decay_epsilon()
        
        # Guardar estadísticas
        rewards_per_episode.append(total_reward)
        wins_per_episode.append(1 if total_reward > 0 else 0)
        epsilon_history.append(agent.epsilon)
        
        # Actualizar barra de progreso con información
        if verbose and episode % 100 == 0:
            recent_wins = np.mean(wins_per_episode[-100:]) * 100 if len(wins_per_episode) >= 100 else 0
            iterator.set_postfix({
                'Win Rate (últimos 100)': f'{recent_wins:.1f}%',
                'Epsilon': f'{agent.epsilon:.3f}'
            })
    
    return {
        'rewards': rewards_per_episode,
        'wins': wins_per_episode,
        'epsilon_history': epsilon_history
    }


def evaluate_agent(env, agent, n_episodes=10):
    wins = 0
    episode_details = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        trajectory = [(state, None, None)]
        
        while not done:
            # Elegir acción de forma codiciosa (sin exploración)
            action = agent.choose_action(state, training=False)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trajectory.append((next_state, action, reward))
            
            state = next_state
            total_reward += reward
            steps += 1
        
        if total_reward > 0:
            wins += 1
        
        episode_details.append({
            'episode': episode + 1,
            'win': total_reward > 0,
            'reward': total_reward,
            'steps': steps,
            'trajectory': trajectory
        })
    
    win_rate = (wins / n_episodes) * 100
    
    return {
        'win_rate': win_rate,
        'wins': wins,
        'episodes': episode_details
    }
