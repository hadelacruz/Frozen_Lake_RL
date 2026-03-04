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

### Opción 1: Ejecutar el Notebook (Recomendado)

1. Abre `results.ipynb` en Jupyter o VS Code
2. Haz clic en **"Run All"** (Ejecutar Todo)

El notebook automáticamente:
- ✅ Instala todas las dependencias necesarias
- ✅ Entrena el agente Q-Learning (10,000 episodios) si no hay resultados previos
- ✅ Carga resultados existentes si ya fueron entrenados
- ✅ Muestra todas las visualizaciones y análisis

### Task 2 - Lab 5 Completo

El notebook ejecuta las 3 tareas del Lab 5:

**Task 2.1:** Inicialización del entorno Frozen Lake (is_slippery=True)
**Task 2.2:** Entrenamiento con Q-Learning (10,000 episodios, epsilon-greedy)
**Task 2.3:** Evaluación del agente (10 episodios, cálculo de win rate)

### Instalación Manual (Opcional)

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
- ✅ Configuración y verificación del entorno
- ✅ Hiperparámetros del algoritmo Q-Learning
- ✅ Progreso del entrenamiento (gráficas y estadísticas)
- ✅ Tabla Q aprendida (heatmap visual)
- ✅ Win Rate y análisis de evaluación
- ✅ Visualización de episodios exitosos
- ✅ Conclusiones y análisis de resultados

## Tecnologías Utilizadas

- **Python 3.13**
- **Gymnasium** - Framework de Reinforcement Learning
- **NumPy** - Manejo de la tabla Q
- **Matplotlib** - Visualizaciones
- **Jupyter Notebook** - Presentación de resultados

---

**Proyecto completado para Lab 5 - Inteligencia Artificial** ✓

