import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def calcular_histograma(imagen):
    # Calcula el histograma de la imagen en escala de grises
    histograma, _ = np.histogram(imagen.ravel(), bins=256, range=(0, 256))
    return histograma

def calcular_entropia(histograma, total_pixeles):
    # Cálculo de la entropía
    probabilidad = histograma / total_pixeles
    probabilidad = probabilidad[probabilidad > 0]  # Filtrar ceros
    entropia = -np.sum(probabilidad * np.log2(probabilidad))
    return entropia

def encontrar_mejor_umbral_entropia(imagen):
    # Calcular el histograma de la imagen
    histograma = calcular_histograma(imagen)
    total_pixeles = imagen.size
    
    # Variables para encontrar el mejor umbral
    mejor_entropia = 0
    mejor_umbral = 0

    # Calcular la entropía para cada posible umbral
    for umbral in range(256):
        hist_bajo = histograma[:umbral]
        hist_alto = histograma[umbral:]
        
        # Cálculo de la entropía para las dos clases
        entropia_bajo = calcular_entropia(hist_bajo, total_pixeles)
        entropia_alto = calcular_entropia(hist_alto, total_pixeles)
        
        # Entropía total
        entropia_total = entropia_bajo + entropia_alto
        
        if entropia_total > mejor_entropia:
            mejor_entropia = entropia_total
            mejor_umbral = umbral
    
    return mejor_umbral

def aplicar_tonos(imagen, num_tonos):
    # Definir los umbrales
    max_val = 255
    umbrales = np.linspace(0, max_val, num_tonos + 1)
    imagen_tonos = np.zeros_like(imagen)

    # Asignar tonos basados en los umbrales definidos
    for i in range(num_tonos):
        lower_bound = umbrales[i]
        upper_bound = umbrales[i + 1]
        imagen_tonos[(imagen >= lower_bound) & (imagen < upper_bound)] = int((lower_bound + upper_bound) / 2)
    
    return imagen_tonos

# Cargar la imagen
imagen_a_color = cv2.imread("imagen a color.png")
if imagen_a_color is None:
    print("La imagen no se cargo correctamente, favor de verificar su ruta")
else:
    imagen = cv2.cvtColor(imagen_a_color,cv2.COLOR_BGR2GRAY)

    # Calcular el histograma de la imagen original
    histograma = calcular_histograma(imagen)

    # Calcular el mejor umbral usando el método de entropía
    mejor_umbral = encontrar_mejor_umbral_entropia(imagen)

    # Aplicar diferentes tonos (3, 4, 8, 16)
    tonos = [3, 4, 8, 16]
    imagenes_tonos = [aplicar_tonos(imagen, num_tonos) for num_tonos in tonos]

    # Mostrar el histograma y las imágenes umbralizadas
    plt.figure(figsize=(15, 10))

    # Mostrar el histograma original
    plt.subplot(2, len(tonos), 1)
    plt.bar(range(256), histograma, color='black')
    plt.title('Histograma Original')
    plt.xlim([0, 255])

    # Mostrar las imágenes umbralizadas y guardar las matrices
    for idx, num_tonos in enumerate(tonos):
        plt.subplot(2, len(tonos), idx + 2)
        plt.imshow(imagenes_tonos[idx], cmap='gray')
        
        # Guardar la matriz de cada imagen en un archivo CSV
        nombre_archivo = f"imagen_{num_tonos}_tonos_entropia.csv"
        np.savetxt(nombre_archivo, imagenes_tonos[idx], delimiter=',', fmt='%d')
        
        plt.title(f'{num_tonos} tonos')

    for num_tonos in tonos:
        imagen_tonos = aplicar_tonos(imagen, num_tonos)
        # Guardar la imagen umbralizada con diferentes tonos
        cv2.imwrite(f'imagen_{num_tonos}_tonos.png', imagen_tonos)

    # Mostrar las imágenes y el histograma
    plt.tight_layout()
    plt.show()
