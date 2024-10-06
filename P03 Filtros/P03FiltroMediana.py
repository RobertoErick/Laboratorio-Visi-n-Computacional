import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funcion para crea la matriz de dispersion de la imagen original en escala de grises
def crear_matriz_dispersion(imagen, promedio):
    nueva_matriz_de_dispersion = imagen.copy()

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            
            nueva_matriz_de_dispersion[i][j] = abs(imagen[i,j] - promedio) * 100 / promedio

    return nueva_matriz_de_dispersion

# Funcion para crear la matriz de la imagen original en escala de grises
def histograma_imagen(imagen):
    plt.hist(imagen.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)

    plt.title('Histograma de Barras - Imagen Original')
    plt.xlabel('Valor de Intensidad (0-255)')
    plt.ylabel('Número de Píxeles')

    plt.show()

# Función para crear la matriz de la dispersion de los pixeles
def histograma_matriz_dispersion(matriz_de_dispersion):

    # Histograma de la matriz de dispersion
    plt.hist(matriz_de_dispersion.ravel(), bins=100, range=[0, 100], color='gray', alpha=0.7)

    plt.suptitle('Histograma de Barras - matriz de dispersion mediana')
    plt.title('A continuacion, ingresa el valor del porcentaje para ser reemplazadio por la mediana: %i' %mediana)
    plt.xlabel('Valor de Intensidad (0-255)')
    plt.ylabel('Número de Píxeles')

    plt.show()

# Crear la imagen final del filtrado de imagen con la mediana
def crear_imagen_filtrada(imagen, matriz_de_dispersion, filtro, imagen_filtrada_con_mediana, mediana):
    # Esta funcion de OpenCv ya esta hecha para quitar el ruido de la imagen pero no encaja con los metodos enseñados
    #matriz_de_dispersion = cv2.medianBlur(imagen, 5)

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if matriz_de_dispersion[i][j] >= filtro:
                imagen_filtrada_con_mediana[i][j] = mediana

# Cargar la imagen en escala de grises
imagen_color = cv2.imread("imagen.png")
if imagen_color is None:
    print("Error: No se pudo cargar la imagen")
else:
    # Convertir la imagen a escala de grises
    imagen = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    print("Dimensiones de la imagen en escala de grises:", imagen.shape)
    print("                                                 Y    X")
    
    # Leer las coordenadas de inicio
    x_inicio = int(input("Escoja la posición en X (columna): "))
    y_inicio = int(input("Escoja la posición en Y (fila): "))

    # Definir las dimensiones del recorte (62 en Y, 40 en X)
    x_final = x_inicio + 40
    y_final = y_inicio + 62

    # Asegurarse de que los límites no excedan el tamaño de la imagen
    if x_final > imagen.shape[1] or y_final > imagen.shape[0]:
        print("Error: Las coordenadas exceden el tamaño de la imagen.")
    else:
        # Recortar la imagen utilizando slicing de NumPy
        imagen_recortada = imagen[y_inicio:y_final, x_inicio:x_final]

        # Crear diferentes imagenes para mostrar el resultado
        imagen_filtrada_con_mediana = imagen_recortada.copy()
        matriz_de_dispersion = imagen_recortada.copy()

        # Promedio de la imagen
        promedio = np.mean(imagen_recortada)

        # Mediana de la imagen
        mediana = np.median(imagen_recortada)

        # Creacion de la matriz de dispersion
        matriz_de_dispersion = crear_matriz_dispersion(imagen_recortada, promedio)

        # Histograma de la imagen original en escala de grises
        histograma_imagen(imagen_recortada)

        # Crear el histograma de la matriz dispersion con informacion para el usuario
        histograma_matriz_dispersion(matriz_de_dispersion)

        # Filtro que se le va a colocar a la imagen (a partir de qué número de la matriz de dispersion va a reemplazar los valores en la imagen)
        print("Mediana de la imagen (valor por el que va a cambiar los pixeles seleccionados): ", mediana)
        print("Valor recomendado con las imagenes de prueba: 97")
        filtro = int(input("Introduce el filtro que quieres colocar (0 - 100): "))

        # Creacion de la imagen dependiendo de los datos obtenidos anteriormente y la respuesta del usuario
        crear_imagen_filtrada(imagen_recortada, matriz_de_dispersion, filtro, imagen_filtrada_con_mediana, mediana)

        # Promedio de la imagen filtrada resultante
        promedio_final = np.mean(imagen_filtrada_con_mediana)

        # Obtener la matriz de dispersion resultante
        matriz_de_dispersion_resultante = crear_matriz_dispersion(imagen_filtrada_con_mediana, promedio_final)

        # Dispersion de la matriz resultante final
        print("El valor final de la sumatoria de la matriz resultante es la siguiente: ", np.sum(matriz_de_dispersion_resultante))

        # Guardar matrizes de imágenes procesadas
        np.savetxt('imagen_original.csv', imagen_recortada, delimiter=',', fmt='%d')
        np.savetxt('matriz_de_dispersion_mediana.csv', matriz_de_dispersion, delimiter=',', fmt='%d')
        np.savetxt('imagen_filtrada_con_mediana.csv', imagen_filtrada_con_mediana, delimiter=',', fmt='%d')
        np.savetxt('matriz_de_dispersion_mediana_resultante.csv', matriz_de_dispersion_resultante, delimiter=',', fmt='%d')

        # Mostrar imagenes procesadas
        cv2.imshow("Imagen original", imagen_recortada)
        cv2.imshow("Matriz de dispersión mediana", matriz_de_dispersion)
        cv2.imshow("Imagen filtrada con mediana", imagen_filtrada_con_mediana)
        cv2.imshow("Matriz de dispersion resultante", matriz_de_dispersion_resultante)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Guarar imagenes en los archivos
        cv2.imwrite("imagen_filtrada_con_mediana.png", imagen_filtrada_con_mediana)
