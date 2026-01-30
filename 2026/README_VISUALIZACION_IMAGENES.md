# Visualización de Imágenes de Cáncer de Mama Basada en Píxeles

Este conjunto de scripts permite descargar y visualizar imágenes reales de cáncer de mama basándose en sus valores de píxeles.

## Scripts Disponibles

### 1. `descargar_y_visualizar_imagenes.py`
Script principal que:
- Genera imágenes sintéticas basadas en las características del dataset de Wisconsin
- Visualiza las imágenes basándose en sus valores de píxeles
- Muestra análisis estadístico de los píxeles
- Genera visualizaciones múltiples

### 2. `descargar_dataset_real.py`
Script auxiliar que:
- Verifica si existe un dataset local de imágenes reales
- Proporciona información sobre cómo descargar el dataset BreakHis
- Crea la estructura de directorios necesaria

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Opción 1: Generar y Visualizar Imágenes Sintéticas

```bash
python descargar_y_visualizar_imagenes.py
```

Este script:
1. Genera imágenes sintéticas basadas en las 30 características del dataset de Wisconsin
2. Cada imagen representa visualmente las características de una muestra
3. Las imágenes se organizan en carpetas por clase (benign/malignant)
4. Visualiza una imagen aleatoria con análisis detallado de píxeles
5. Muestra múltiples imágenes en una cuadrícula

### Opción 2: Usar Dataset Real (BreakHis)

1. **Descargar BreakHis manualmente:**
   - Visita: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
   - Completa el formulario de registro
   - Descarga el archivo ZIP
   - Extrae las imágenes en `breast_cancer_images_real/`

2. **Estructura esperada:**
```
breast_cancer_images_real/
├── benign/
│   ├── SOB_B_*.png
│   └── ...
└── malignant/
    ├── SOB_M_*.png
    └── ...
```

3. **Verificar dataset:**
```bash
python descargar_dataset_real.py
```

4. **Modificar el script principal** para usar el dataset real en lugar del sintético.

## Características de la Visualización

El script genera visualizaciones que incluyen:

1. **Imagen Original**: Muestra la imagen tal como está almacenada en píxeles
2. **Histograma de Píxeles**: Distribución de valores de píxeles (0-255)
3. **Estadísticas**: Análisis estadístico completo de los valores de píxeles
4. **Mapa de Calor**: Visualización de los valores de píxeles como mapa de calor

## Salidas Generadas

- `visualizacion_pixeles_*.png`: Visualización detallada de una imagen
- `visualizacion_multiples_imagenes.png`: Cuadrícula con múltiples imágenes
- `breast_cancer_images/`: Directorio con las imágenes generadas

## Notas Importantes

- El dataset de Wisconsin original NO contiene imágenes de píxeles, solo características numéricas
- Las imágenes sintéticas generadas representan visualmente las características numéricas
- Para imágenes reales de histopatología, se recomienda usar el dataset BreakHis
- Las imágenes sintéticas son útiles para entender cómo se visualizarían los datos como imágenes

## Requisitos

- Python 3.7+
- numpy
- matplotlib
- Pillow (PIL)
- scikit-learn
- pandas
- seaborn

## Referencias

- **BreakHis Dataset**: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- **Wisconsin Dataset**: https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

