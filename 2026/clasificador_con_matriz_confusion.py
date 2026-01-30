"""
================================================================================
EJEMPLO EDUCATIVO: CLASIFICADOR CON MATRIZ DE CONFUSIÓN
================================================================================
Autor: Profesor de Machine Learning
Objetivo: Demostrar el uso de un clasificador clásico en scikit-learn y cómo
          interpretar los resultados mediante una matriz de confusión.

Este ejemplo utiliza el dataset de Cáncer de Mama de Wisconsin (Breast Cancer),
uno de los datasets más utilizados para enseñar clasificación binaria.

Contenido:
1. Carga y exploración del dataset
2. Preprocesamiento de datos
3. División en conjuntos de entrenamiento y prueba
4. Entrenamiento del modelo (Random Forest)
5. Evaluación con matriz de confusión
6. Interpretación de métricas

================================================================================
"""

# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configuración para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def cargar_y_explorar_datos():
    """
    Carga el dataset de cáncer de mama y muestra información básica.
    
    El dataset contiene:
    - 569 muestras
    - 30 características (features) numéricas
    - 2 clases: maligno (0) y benigno (1)
    """
    print("=" * 70)
    print("1. CARGA Y EXPLORACIÓN DEL DATASET")
    print("=" * 70)
    
    # Cargar el dataset
    datos = load_breast_cancer()
    
    X = datos.data      # Características (features)
    y = datos.target    # Etiquetas (labels)
    
    print(f"\n📊 Dataset: {datos.DESCR.split(chr(10))[0]}")
    print(f"\n📌 Dimensiones de X (características): {X.shape}")
    print(f"   → {X.shape[0]} muestras")
    print(f"   → {X.shape[1]} características por muestra")
    
    print(f"\n📌 Clases del problema:")
    for i, nombre_clase in enumerate(datos.target_names):
        conteo = np.sum(y == i)
        porcentaje = (conteo / len(y)) * 100
        print(f"   → Clase {i} ({nombre_clase}): {conteo} muestras ({porcentaje:.1f}%)")
    
    print(f"\n📌 Primeras 5 características:")
    for i, nombre in enumerate(datos.feature_names[:5]):
        print(f"   {i+1}. {nombre}")
    print("   ...")
    
    return X, y, datos.target_names


def preprocesar_datos(X, y):
    """
    Preprocesa los datos:
    1. Divide en conjuntos de entrenamiento (80%) y prueba (20%)
    2. Escala las características usando StandardScaler
    
    ¿Por qué escalar?
    - Muchos algoritmos funcionan mejor cuando las características
      están en la misma escala.
    - StandardScaler transforma los datos para que tengan media 0
      y desviación estándar 1.
    """
    print("\n" + "=" * 70)
    print("2. PREPROCESAMIENTO DE DATOS")
    print("=" * 70)
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,        # 20% para prueba
        random_state=RANDOM_STATE,
        stratify=y             # Mantener proporción de clases
    )
    
    print(f"\n📌 División de datos:")
    print(f"   → Entrenamiento: {X_train.shape[0]} muestras ({100*X_train.shape[0]/len(y):.0f}%)")
    print(f"   → Prueba: {X_test.shape[0]} muestras ({100*X_test.shape[0]/len(y):.0f}%)")
    
    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar
    X_test_scaled = scaler.transform(X_test)        # Solo transformar
    
    print(f"\n📌 Escalado aplicado (StandardScaler):")
    print(f"   → Media antes: {X_train[:, 0].mean():.2f}")
    print(f"   → Media después: {X_train_scaled[:, 0].mean():.2f}")
    print(f"   → Desv. Est. antes: {X_train[:, 0].std():.2f}")
    print(f"   → Desv. Est. después: {X_train_scaled[:, 0].std():.2f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def entrenar_modelo(X_train, y_train):
    """
    Entrena un clasificador Random Forest.
    
    ¿Por qué Random Forest?
    - Es un algoritmo de ensemble que combina múltiples árboles de decisión
    - Robusto ante overfitting
    - No requiere mucho ajuste de hiperparámetros
    - Funciona bien con datos de alta dimensionalidad
    """
    print("\n" + "=" * 70)
    print("3. ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    
    print("\n📌 Algoritmo: Random Forest Classifier")
    print("   → Tipo: Ensemble de árboles de decisión")
    print("   → Hiperparámetros:")
    
    modelo = RandomForestClassifier(
        n_estimators=100,           # Número de árboles
        max_depth=10,               # Profundidad máxima
        min_samples_split=5,        # Mínimo de muestras para dividir
        random_state=RANDOM_STATE
    )
    
    print(f"      • n_estimators: 100 (número de árboles)")
    print(f"      • max_depth: 10 (profundidad máxima)")
    print(f"      • min_samples_split: 5")
    
    # Entrenamiento
    print("\n🔄 Entrenando modelo...")
    modelo.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente!")
    
    return modelo


def evaluar_modelo(modelo, X_test, y_test, nombres_clases):
    """
    Evalúa el modelo entrenado y genera la matriz de confusión.
    
    Métricas importantes:
    - Accuracy: Proporción de predicciones correctas
    - Precision: De los positivos predichos, ¿cuántos son realmente positivos?
    - Recall (Sensibilidad): De los positivos reales, ¿cuántos detectamos?
    - F1-Score: Media armónica entre Precision y Recall
    """
    print("\n" + "=" * 70)
    print("4. EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    # Realizar predicciones
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n📊 MÉTRICAS DE EVALUACIÓN:")
    print("-" * 40)
    print(f"   Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}  ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f}  ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f}  ({f1*100:.2f}%)")
    
    # Reporte de clasificación detallado
    print("\n📋 REPORTE DE CLASIFICACIÓN DETALLADO:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, cm


def mostrar_matriz_confusion(y_test, y_pred, nombres_clases):
    """
    Visualiza la matriz de confusión con una explicación detallada.
    
    La matriz de confusión muestra:
    - Verdaderos Positivos (VP): Predichos como positivos y son positivos
    - Verdaderos Negativos (VN): Predichos como negativos y son negativos  
    - Falsos Positivos (FP): Predichos como positivos pero son negativos
    - Falsos Negativos (FN): Predichos como negativos pero son positivos
    """
    print("\n" + "=" * 70)
    print("5. MATRIZ DE CONFUSIÓN")
    print("=" * 70)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n📊 MATRIZ DE CONFUSIÓN (valores):")
    print("-" * 40)
    print(f"\n                    PREDICCIÓN")
    print(f"                 {nombres_clases[0]:^10} {nombres_clases[1]:^10}")
    print(f"              ┌───────────┬───────────┐")
    print(f"    REAL      │           │           │")
    print(f"  {nombres_clases[0]:^10} │  {cm[0,0]:^7}  │  {cm[0,1]:^7}  │")
    print(f"              ├───────────┼───────────┤")
    print(f"  {nombres_clases[1]:^10} │  {cm[1,0]:^7}  │  {cm[1,1]:^7}  │")
    print(f"              └───────────┴───────────┘")
    
    # Interpretación
    print("\n📖 INTERPRETACIÓN:")
    print("-" * 40)
    
    # Para clasificación binaria de cáncer: 0=maligno, 1=benigno
    vn = cm[0, 0]  # Verdaderos Negativos (maligno correcto)
    fp = cm[0, 1]  # Falsos Positivos (maligno predicho como benigno) - ¡PELIGROSO!
    fn = cm[1, 0]  # Falsos Negativos (benigno predicho como maligno)
    vp = cm[1, 1]  # Verdaderos Positivos (benigno correcto)
    
    print(f"\n   ✅ Verdaderos Negativos (VN): {vn}")
    print(f"      → Casos malignos correctamente identificados")
    
    print(f"\n   ✅ Verdaderos Positivos (VP): {vp}")
    print(f"      → Casos benignos correctamente identificados")
    
    print(f"\n   ❌ Falsos Positivos (FP): {fp}")
    print(f"      → Casos malignos clasificados como benignos")
    print(f"      → ¡Error crítico en diagnóstico médico!")
    
    print(f"\n   ❌ Falsos Negativos (FN): {fn}")
    print(f"      → Casos benignos clasificados como malignos")
    print(f"      → Genera ansiedad innecesaria al paciente")
    
    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de confusión con valores absolutos
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=nombres_clases
    )
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Matriz de Confusión\n(Valores Absolutos)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicción', fontsize=11)
    axes[0].set_ylabel('Valor Real', fontsize=11)
    
    # Matriz de confusión normalizada (porcentajes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=nombres_clases
    )
    disp2.plot(ax=axes[1], cmap='Greens', values_format='.2%')
    axes[1].set_title('Matriz de Confusión\n(Normalizada por Fila)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicción', fontsize=11)
    axes[1].set_ylabel('Valor Real', fontsize=11)
    
    plt.suptitle('Evaluación del Clasificador Random Forest\nDataset: Breast Cancer Wisconsin', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n💾 Gráfica guardada como 'matriz_confusion.png'")


def main():
    """
    Función principal que ejecuta todo el pipeline de clasificación.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  CLASIFICADOR DE MACHINE LEARNING CON MATRIZ DE CONFUSIÓN  ".center(68) + "█")
    print("█" + "  Dataset: Breast Cancer Wisconsin  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Paso 1: Cargar datos
    X, y, nombres_clases = cargar_y_explorar_datos()
    
    # Paso 2: Preprocesar
    X_train, X_test, y_train, y_test = preprocesar_datos(X, y)
    
    # Paso 3: Entrenar
    modelo = entrenar_modelo(X_train, y_train)
    
    # Paso 4: Evaluar
    y_pred, cm = evaluar_modelo(modelo, X_test, y_test, nombres_clases)
    
    # Paso 5: Visualizar matriz de confusión
    mostrar_matriz_confusion(y_test, y_pred, nombres_clases)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("6. RESUMEN Y CONCLUSIONES")
    print("=" * 70)
    print("""
    📝 PUNTOS CLAVE DEL EJERCICIO:
    
    1. La matriz de confusión es fundamental para entender el comportamiento
       de un clasificador más allá de la simple accuracy.
    
    2. En problemas médicos, los Falsos Negativos (FN) pueden ser más
       críticos que los Falsos Positivos (FP).
    
    3. El preprocesamiento (escalado) mejora el rendimiento de muchos
       algoritmos de Machine Learning.
    
    4. Random Forest es un buen punto de partida por su robustez y
       facilidad de uso.
    
    5. Siempre dividir los datos en entrenamiento y prueba para
       evaluar la capacidad de generalización del modelo.
    """)
    
    print("=" * 70)
    print("FIN DEL EJEMPLO")
    print("=" * 70)


if __name__ == "__main__":
    main()


