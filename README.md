# Frozen Lake Q-Learning - Lab 5

Proyecto de Inteligencia Artificial - Implementación de Q-Learning para resolver el entorno Frozen Lake.

## Estructura del Proyecto

```
Frozen_Lake_RL/
├── src/
│   ├── __init__.py          # Inicialización del paquete
│   ├── environment.py       # Gestión del entorno Frozen Lake
│   ├── agent.py             # Agente Q-Learning con Epsilon-Greedy
│   ├── q_learning.py        # Algoritmo de entrenamiento y evaluación
│   └── utils.py             # Utilidades de visualización y análisis
├── results.ipynb            # Notebook principal (TODO EN UNO)
├── results/                 # Directorio con resultados guardados
│   ├── q_table.npy
│   ├── training_stats.pkl
│   ├── evaluation_results.pkl
│   ├── training_progress.png
│   └── q_table_heatmap.png
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

## Uso

### Opción 1: Ejecutar el Notebook

1. Abre `results.ipynb` en Jupyter o VS Code
2. Haz clic en **"Run All"** (Ejecutar Todo)

### Instalación Manual 

Si prefieres instalar dependencias manualmente antes de ejecutar el notebook:

```bash
pip install -r requirements.txt
```

### Re-entrenar desde Cero

Si deseas volver a entrenar el agente:
1. Elimina la carpeta `results/`
2. Ejecuta el notebook nuevamente

## Contenido del Notebook

El notebook `results.ipynb` muestra:
- Configuración y verificación del entorno
- Hiperparámetros del algoritmo Q-Learning
- Progreso del entrenamiento (gráficas y estadísticas)
- Tabla Q aprendida (heatmap visual)
- Win Rate y análisis de evaluación
- Visualización de episodios exitosos
- Conclusiones y análisis de resultados

