"""
Visualización Gráfica de una Muestra Aleatoria del Dataset Breast Cancer Wisconsin

Este script carga el dataset y visualiza gráficamente las características
de una muestra aleatoria seleccionada.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Configuración
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Cargar el dataset
datos = load_breast_cancer()
X = datos.data
y = datos.target
nombres_features = datos.feature_names
nombres_clases = datos.target_names

# Seleccionar una muestra aleatoria
indice_aleatorio = np.random.randint(0, len(X))
muestra = X[indice_aleatorio]
clase_muestra = y[indice_aleatorio]
nombre_clase = nombres_clases[clase_muestra]

print(f"Muestra seleccionada: índice {indice_aleatorio}")
print(f"Clase: {nombre_clase}")

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Gráfico de barras horizontales (todas las características)
ax1 = fig.add_subplot(gs[0, :])
y_pos = np.arange(len(nombres_features))
colores = ['#e74c3c' if clase_muestra == 0 else '#27ae60'] * len(nombres_features)
ax1.barh(y_pos, muestra, color=colores, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(nombres_features, fontsize=8)
ax1.set_xlabel('Valor de la Característica', fontsize=11, fontweight='bold')
ax1.set_title(f'Visualización de Todas las Características - Muestra #{indice_aleatorio} ({nombre_clase.upper()})', 
              fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 2. Gráfico de barras verticales (primeras 10 características)
ax2 = fig.add_subplot(gs[1, 0])
features_principales = nombres_features[:10]
valores_principales = muestra[:10]
x_pos = np.arange(len(features_principales))
ax2.bar(x_pos, valores_principales, color=colores[0], alpha=0.7, edgecolor='black', linewidth=1)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(features_principales, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Valor', fontsize=10, fontweight='bold')
ax2.set_title('Primeras 10 Características', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Gráfico de líneas (todas las características)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(muestra, marker='o', linewidth=2, markersize=4, color=colores[0], alpha=0.7)
ax3.set_xlabel('Índice de Característica', fontsize=10, fontweight='bold')
ax3.set_ylabel('Valor', fontsize=10, fontweight='bold')
ax3.set_title('Perfil de Características (Línea)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, len(nombres_features), 5))

# 4. Gráfico de barras agrupadas por tipo de característica
ax4 = fig.add_subplot(gs[2, 0])
# Agrupar características por tipo (mean, se, worst)
tipos = ['mean', 'se', 'worst']
colores_tipos = ['#3498db', '#9b59b6', '#f39c12']
ancho = 0.25
x = np.arange(10)  # Primeras 10 características base

for i, tipo in enumerate(tipos):
    indices_tipo = [j for j, feat in enumerate(nombres_features[:30]) if tipo in feat][:10]
    valores_tipo = [muestra[j] for j in indices_tipo[:10]]
    if len(valores_tipo) == 10:
        ax4.bar(x + i*ancho, valores_tipo, ancho, label=tipo, color=colores_tipos[i], alpha=0.7)

ax4.set_xlabel('Características Base', fontsize=10, fontweight='bold')
ax4.set_ylabel('Valor', fontsize=10, fontweight='bold')
ax4.set_title('Características Agrupadas por Tipo (mean/se/worst)', fontsize=11, fontweight='bold')
ax4.set_xticks(x + ancho)
ax4.set_xticklabels([feat.replace(' mean', '').replace(' se', '').replace(' worst', '') 
                     for feat in nombres_features[:10]], rotation=45, ha='right', fontsize=8)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Heatmap de características organizadas
ax5 = fig.add_subplot(gs[2, 1])
# Organizar características en una matriz 6x5 para visualización
matriz_caracteristicas = muestra.reshape(6, 5)
im = ax5.imshow(matriz_caracteristicas, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
ax5.set_title('Mapa de Calor de Características', fontsize=11, fontweight='bold')
ax5.set_xlabel('Grupo 1', fontsize=9)
ax5.set_ylabel('Grupo 2', fontsize=9)
plt.colorbar(im, ax=ax5, label='Valor')

# Título general
fig.suptitle(f'Visualización Completa de la Muestra #{indice_aleatorio} - Clase: {nombre_clase.upper()}', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('visualizacion_muestra_aleatoria.png', dpi=150, bbox_inches='tight')
print("\nVisualización guardada como 'visualizacion_muestra_aleatoria.png'")
plt.show()

