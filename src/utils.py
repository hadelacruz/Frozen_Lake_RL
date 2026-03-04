import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def plot_training_progress(training_stats, save_path=None):
    rewards = training_stats['rewards']
    wins = training_stats['wins']
    epsilon_history = training_stats['epsilon_history']
    
    # Calcular win rate en ventanas de 100 episodios
    window = 100
    win_rates = []
    for i in range(len(wins)):
        start = max(0, i - window + 1)
        win_rates.append(np.mean(wins[start:i+1]) * 100)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfica 1: Win Rate
    axes[0].plot(win_rates, linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Episodio', fontsize=12)
    axes[0].set_ylabel('Win Rate (%)', fontsize=12)
    axes[0].set_title('Win Rate Durante el Entrenamiento (Ventana de 100 episodios)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])
    
    # Gráfica 2: Epsilon Decay
    axes[1].plot(epsilon_history, linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Episodio', fontsize=12)
    axes[1].set_ylabel('Epsilon (ε)', fontsize=12)
    axes[1].set_title('Decaimiento de Epsilon Durante el Entrenamiento', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    return fig


def plot_q_table_heatmap(q_table, save_path=None):
    action_names = ['Left', 'Down', 'Right', 'Up']
    
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(q_table, cmap='viridis', aspect='auto')
    
    # Etiquetas
    ax.set_xticks(np.arange(len(action_names)))
    ax.set_yticks(np.arange(q_table.shape[0]))
    ax.set_xticklabels(action_names)
    ax.set_yticklabels([f'S{i}' for i in range(q_table.shape[0])])
    
    # Títulos
    ax.set_xlabel('Acciones', fontsize=12)
    ax.set_ylabel('Estados', fontsize=12)
    ax.set_title('Tabla Q - Valores Aprendidos', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valor Q', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    return fig


def visualize_episode(episode_data):
    trajectory = episode_data['trajectory']
    action_names = {0: 'Left ←', 1: 'Down ↓', 2: 'Right →', 3: 'Up ↑'}
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"EPISODIO {episode_data['episode']}")
    output.append(f"{'='*60}")
    output.append(f"Resultado: {'VICTORIA ✓' if episode_data['win'] else 'DERROTA ✗'}")
    output.append(f"Recompensa Total: {episode_data['reward']}")
    output.append(f"Número de Pasos: {episode_data['steps']}")
    output.append(f"\nTrayectoria:")
    output.append(f"{'-'*60}")
    
    for i, (state, action, reward) in enumerate(trajectory):
        if action is not None:
            output.append(f"Paso {i}: Estado {state} → Acción: {action_names[action]} → Recompensa: {reward}")
        else:
            output.append(f"Estado Inicial: {state}")
    
    output.append(f"{'='*60}\n")
    
    return '\n'.join(output)


def save_results(agent, training_stats, evaluation_results, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Guardar tabla Q
    q_table_path = os.path.join(save_dir, 'q_table.npy')
    np.save(q_table_path, agent.get_q_table())
    print(f"Tabla Q guardada en: {q_table_path}")
    
    # Guardar estadísticas
    stats_path = os.path.join(save_dir, 'training_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(training_stats, f)
    print(f"Estadísticas de entrenamiento guardadas en: {stats_path}")
    
    # Guardar resultados de evaluación
    eval_path = os.path.join(save_dir, 'evaluation_results.pkl')
    with open(eval_path, 'wb') as f:
        pickle.dump(evaluation_results, f)
    print(f"Resultados de evaluación guardados en: {eval_path}")


def load_results(save_dir='results'):
    q_table = np.load(os.path.join(save_dir, 'q_table.npy'))
    
    with open(os.path.join(save_dir, 'training_stats.pkl'), 'rb') as f:
        training_stats = pickle.load(f)
    
    with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'rb') as f:
        evaluation_results = pickle.load(f)
    
    return q_table, training_stats, evaluation_results


def print_evaluation_summary(evaluation_results):
    print("\n" + "="*70)
    print("RESUMEN DE EVALUACIÓN")
    print("="*70)
    print(f"Número de Episodios: {len(evaluation_results['episodes'])}")
    print(f"Victorias: {evaluation_results['wins']}")
    print(f"Win Rate: {evaluation_results['win_rate']:.2f}%")
    print("="*70 + "\n")
    
    # Detalles por episodio
    for ep in evaluation_results['episodes']:
        resultado = "VICTORIA" if ep['win'] else "DERROTA"
        print(f"Episodio {ep['episode']}: {resultado} - {ep['steps']} pasos - Recompensa: {ep['reward']}")
