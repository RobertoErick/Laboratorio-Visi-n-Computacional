import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('incendio.png')
# imagen = cv2.resize(imagen, (800, 600))  # Redimensionar si es necesario

# Función para crear un gradiente direccional
def crear_gradiente(direccion, tamaño, intensidad):
    if direccion == 'izquierda':
        gradiente = np.tile(np.linspace(intensidad, 0, tamaño[1]), (tamaño[0], 1))
    elif direccion == 'derecha':
        gradiente = np.tile(np.linspace(0, intensidad, tamaño[1]), (tamaño[0], 1))
    elif direccion == 'arriba':
        gradiente = np.tile(np.linspace(intensidad, 0, tamaño[0]), (tamaño[1], 1)).T
    elif direccion == 'abajo':
        gradiente = np.tile(np.linspace(0, intensidad, tamaño[0]), (tamaño[1], 1)).T
    
    return cv2.merge([gradiente, gradiente, gradiente])

# Aplicar el gradiente de luz a la imagen
def aplicar_gradiente(imagen, gradiente):
    return cv2.addWeighted(imagen, 1.0, gradiente.astype(np.uint8), 0.5, 0)

# Direccion inicial de la luz
direccion_luz = 'izquierda'
gradiente = crear_gradiente(direccion_luz, imagen.shape[:2], 255)
resultado = aplicar_gradiente(imagen, gradiente)

# Mostrar la imagen inicial
cv2.imshow('Efecto de Luz con Gradiente', resultado)

# Cambiar la dirección de la luz
while True:
    tecla = cv2.waitKey(10)
    
    if tecla == 27:  # Esc para salir
        break
    elif tecla == ord('a'):  # Izquierda
        direccion_luz = 'izquierda'
    elif tecla == ord('d'):  # Derecha
        direccion_luz = 'derecha'
    elif tecla == ord('w'):  # Arriba
        direccion_luz = 'arriba'
    elif tecla == ord('s'):  # Abajo
        direccion_luz = 'abajo'

    # Actualizar el gradiente de luz
    gradiente = crear_gradiente(direccion_luz, imagen.shape[:2], 255)
    
    # Aplicar el gradiente de luz y mostrar el resultado
    resultado = aplicar_gradiente(imagen, gradiente)
    cv2.imshow('Efecto de Luz con Gradiente', resultado)

cv2.destroyAllWindows()