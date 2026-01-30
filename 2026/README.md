# 🎓 Ejemplo Educativo: Clasificador con Matriz de Confusión

## Descripción

Este ejemplo demuestra el uso de un **clasificador Random Forest** utilizando scikit-learn, con énfasis en la interpretación de resultados mediante una **matriz de confusión**.

## 📚 Conceptos Cubiertos

1. **Carga de datos** - Uso de datasets incluidos en scikit-learn
2. **Preprocesamiento** - División train/test y escalado con StandardScaler
3. **Entrenamiento** - Random Forest Classifier
4. **Evaluación** - Métricas de clasificación
5. **Visualización** - Matriz de confusión

## 🔬 Dataset Utilizado

**Breast Cancer Wisconsin Dataset**
- 569 muestras
- 30 características numéricas
- 2 clases: maligno (0) y benigno (1)

## 📊 La Matriz de Confusión

```
                    PREDICCIÓN
                 Negativo   Positivo
              ┌───────────┬───────────┐
    REAL      │           │           │
  Negativo    │    VN     │    FP     │
              ├───────────┼───────────┤
  Positivo    │    FN     │    VP     │
              └───────────┴───────────┘
```

- **VN (Verdaderos Negativos)**: Correctamente clasificados como negativos
- **VP (Verdaderos Positivos)**: Correctamente clasificados como positivos
- **FP (Falsos Positivos)**: Negativos clasificados erróneamente como positivos
- **FN (Falsos Negativos)**: Positivos clasificados erróneamente como negativos

## 📐 Métricas Derivadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Accuracy** | (VP + VN) / Total | Proporción de aciertos totales |
| **Precision** | VP / (VP + FP) | De los predichos positivos, ¿cuántos son correctos? |
| **Recall** | VP / (VP + FN) | De los positivos reales, ¿cuántos detectamos? |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance entre Precision y Recall |

## 🚀 Ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el ejemplo
python clasificador_con_matriz_confusion.py
```

## 📦 Dependencias

- numpy
- matplotlib
- scikit-learn

## 📈 Salida Esperada

El script genera:
1. Información detallada del dataset
2. Métricas de evaluación en consola
3. Reporte de clasificación
4. Gráfico de matriz de confusión guardado como `matriz_confusion.png`

## 🎯 Objetivos de Aprendizaje

Después de este ejemplo, deberías poder:

- [ ] Cargar y explorar datasets de scikit-learn
- [ ] Dividir datos en conjuntos de entrenamiento y prueba
- [ ] Entrenar un clasificador Random Forest
- [ ] Interpretar una matriz de confusión
- [ ] Calcular y entender métricas de clasificación

## 📖 Referencias

- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Dataset Breast Cancer Wisconsin](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


