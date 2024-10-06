import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def calcular_histograma(imagen):
    # Calcula el histograma de la imagen en escala de grises
    histograma, _ = np.histogram(imagen.ravel(), bins=256, range=(0, 256))
    return histograma

def calcular_grupo_varianza(histograma, total_pixeles):
    # Cálculo de varianza entre los grupos
    suma_total = np.sum([i * histograma[i] for i in range(256)])
    suma_b = 0
    w_b = 0
    varianza_max = 0
    mejor_umbral = 0

    for umbral in range(256):
        w_b += histograma[umbral]
        w_f = total_pixeles - w_b
        if w_b == 0 or w_f == 0:
            continue
        
        suma_b += umbral * histograma[umbral]
        m_b = suma_b / w_b if w_b != 0 else 0
        m_f = (suma_total - suma_b) / w_f if w_f != 0 else 0

        varianza_entre_clases = w_b * w_f * (m_b - m_f) ** 2

        if varianza_entre_clases > varianza_max:
            varianza_max = varianza_entre_clases
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
    print("La ruta de la imagen no es correcta, porfavor ingrese la ruta adecuada")
else:
    # Convertrir la imagen a escala de grises
    imagen = cv2.cvtColor(imagen_a_color,cv2.COLOR_BGR2GRAY)

    # Calcular el histograma de la imagen original
    histograma = calcular_histograma(imagen)
    total_pixeles = imagen.size

    # Calcular el mejor umbral usando el método de grupo varianza
    mejor_umbral = calcular_grupo_varianza(histograma, total_pixeles)

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

    # Mostrar las imágenes umbralizadas
    for idx, num_tonos in enumerate(tonos):
        plt.subplot(2, len(tonos), idx + 2)
        plt.imshow(imagenes_tonos[idx], cmap='gray')
        
        # Crear nombre de archivo dinámicamente usando f-string
        nombre_archivo = f"imagen_{num_tonos}_tonos.csv"
        np.savetxt(nombre_archivo, imagenes_tonos[idx], delimiter=',', fmt='%d')
        
        plt.title(f'{num_tonos} tonos')

    for num_tonos in tonos:
        imagen_tonos = aplicar_tonos(imagen, num_tonos)
        # Guardar la imagen umbralizada con diferentes tonos
        cv2.imwrite(f'imagen_{num_tonos}_tonos.png', imagen_tonos)

    # Mostrar las imágenes y el histograma
    plt.tight_layout()
    plt.show()
