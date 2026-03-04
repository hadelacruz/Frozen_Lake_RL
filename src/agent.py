"""
Módulo del agente con estrategia Epsilon-Greedy.
"""
import numpy as np


class QLearningAgent:
    """
    Agente que implementa el algoritmo Q-Learning.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            n_states (int): Número de estados en el entorno
            n_actions (int): Número de acciones posibles
            learning_rate (float): Tasa de aprendizaje (α)
            discount_factor (float): Factor de descuento (γ)
            epsilon (float): Valor inicial de epsilon para exploración
            epsilon_min (float): Valor mínimo de epsilon
            epsilon_decay (float): Factor de decaimiento de epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicializar la tabla Q con ceros
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state, training=True):
        """
        Selecciona una acción usando la estrategia Epsilon-Greedy.
        
        Args:
            state (int): Estado actual
            training (bool): Si está en modo entrenamiento (usa epsilon) o evaluación
            
        Returns:
            int: Acción seleccionada
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(self.n_actions)
        else:
            # Explotación: mejor acción según Q-table
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Actualiza el valor Q usando la fórmula de Q-Learning.
        Q(s,a) ← Q(s,a) + α[R + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            reward (float): Recompensa recibida
            next_state (int): Siguiente estado
        """
        # Obtener el valor Q actual
        current_q = self.q_table[state, action]
        
        # Calcular el valor Q máximo del siguiente estado
        max_next_q = np.max(self.q_table[next_state])
        
        # Aplicar la fórmula de Q-Learning
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Actualizar la tabla Q
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """
        Reduce epsilon gradualmente para disminuir la exploración con el tiempo.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_table(self):
        """
        Retorna la tabla Q actual.
        
        Returns:
            np.ndarray: Tabla Q
        """
        return self.q_table
