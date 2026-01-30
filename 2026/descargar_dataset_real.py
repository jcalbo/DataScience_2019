"""
Script para descargar un dataset real de imágenes de cáncer de mama.

Este script intenta descargar imágenes reales de histopatología de cáncer de mama
desde fuentes públicas disponibles.
"""

import os
import requests
import zipfile
from pathlib import Path

DATASET_DIR = "breast_cancer_images_real"
BREAKHIS_URL = "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"


def descargar_breakhis_info():
    """
    Proporciona información sobre cómo descargar el dataset BreakHis.
    BreakHis es un dataset público de imágenes histopatológicas de cáncer de mama.
    """
    print("=" * 70)
    print("INFORMACIÓN SOBRE DATASET BREAKHIS")
    print("=" * 70)
    print("""
    El dataset BreakHis contiene imágenes histopatológicas reales de cáncer de mama.
    
    Para descargarlo:
    1. Visita: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
    2. Completa el formulario de registro
    3. Descarga el archivo ZIP
    4. Extrae las imágenes en el directorio: breast_cancer_images_real/
    
    Estructura esperada:
    breast_cancer_images_real/
    ├── benign/
    │   ├── SOB_B_*.png
    │   └── ...
    └── malignant/
        ├── SOB_M_*.png
        └── ...
    """)
    print("=" * 70)


def verificar_dataset_local():
    """
    Verifica si existe un dataset local de imágenes.
    """
    if os.path.exists(DATASET_DIR):
        subdirs = [d for d in os.listdir(DATASET_DIR) 
                  if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if subdirs:
            print(f"Dataset encontrado en: {os.path.abspath(DATASET_DIR)}")
            print(f"Clases encontradas: {', '.join(subdirs)}")
            
            # Contar imágenes
            total_imagenes = 0
            for subdir in subdirs:
                subdir_path = os.path.join(DATASET_DIR, subdir)
                imagenes = [f for f in os.listdir(subdir_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_imagenes += len(imagenes)
                print(f"  {subdir}: {len(imagenes)} imágenes")
            
            print(f"\nTotal de imágenes: {total_imagenes}")
            return True
    
    print(f"No se encontró dataset en: {DATASET_DIR}")
    return False


def crear_estructura_directorios():
    """
    Crea la estructura de directorios para el dataset.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "benign"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "malignant"), exist_ok=True)
    print(f"Estructura de directorios creada en: {DATASET_DIR}")


def main():
    """
    Función principal.
    """
    print("\nVERIFICANDO DATASET DE IMÁGENES REALES")
    print("=" * 70)
    
    if not verificar_dataset_local():
        print("\nNo se encontró un dataset local.")
        print("\nOpciones:")
        print("1. Descargar BreakHis manualmente (ver instrucciones arriba)")
        print("2. Usar el script descargar_y_visualizar_imagenes.py para generar imágenes sintéticas")
        descargar_breakhis_info()
        crear_estructura_directorios()
    else:
        print("\nDataset listo para usar!")


if __name__ == "__main__":
    main()

