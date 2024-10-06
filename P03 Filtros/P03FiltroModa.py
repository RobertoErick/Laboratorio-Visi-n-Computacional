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

    plt.suptitle('Histograma de Barras - matriz de dispersion moda')
    plt.title('A continuacion, ingresa el valor del porcentaje para ser reemplazado por la moda: %i' %moda)
    plt.xlabel('Valor de Intensidad (0-255)')
    plt.ylabel('Número de Píxeles')

    plt.show()

# Funciones para crear las modas de las imagenes
def imagen_filtrada(imagen):
    for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                if matriz_de_dispersion[i][j] >= filtro :
                    imagen_filtrada_con_moda[i][j] = moda

def imagen_filtrada_multimoda(imagen):
    modas_matrices = []
    modas_sumas = []
    
    # Crear una imagen filtrada por cada moda
    for moda in modas:
        imagen_temp = np.zeros_like(imagen)  # Crear una copia vacía de la imagen
        
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                if matriz_de_dispersion[i][j] >= filtro:
                    imagen_temp[i][j] = moda

        # Calcular la suma de la matriz filtrada
        suma_matriz = np.sum(imagen_temp)
        modas_matrices.append(imagen_temp)
        modas_sumas.append(suma_matriz)

    # Encontrar la moda que genere la suma más pequeña
    indice_menor_suma = np.argmin(modas_sumas)
    imagen_filtrada_con_moda = modas_matrices[indice_menor_suma]
    
    print(f"La moda seleccionada es {modas[indice_menor_suma]} con suma {modas_sumas[indice_menor_suma]}")

    return imagen_filtrada_con_moda

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
        imagen_filtrada_con_moda = imagen_recortada.copy()
        matriz_de_dispersion = imagen_recortada.copy()

        # Promedio de la imagen original
        promedio = np.mean(imagen_recortada)

        # Los siguientes pasos son para obtener la moda o multiples modas
        # Aplanar la imagen recortada
        valores, conteos = np.unique(imagen_recortada.flatten(), return_counts=True)

        # Obtener el valor máximo de frecuencia
        max_frecuencia = np.max(conteos)

        # Obtener todas las modas (valores con la máxima frecuencia)
        modas = valores[conteos == max_frecuencia]

        # Si hay más de una moda
        if len(modas) > 1:
            print(f"Hay múltiples modas: {modas}")
        else:
            print(f"La moda es: {modas[0]}")
            moda = modas[0]
        # Hasta aqui sabemos las modas o moda que se tiene en la imagen

        # Creacion de la matriz de dispersion
        matriz_de_dispersion = crear_matriz_dispersion(imagen_recortada, promedio)
    
        # Histograma de la imagen original en escala de grises
        histograma_imagen(imagen_recortada)

        # Crear el histograma de la matriz dispersion con informacion para el usuario
        histograma_matriz_dispersion(matriz_de_dispersion)

        # Filtro que se le va a colocar a la imagen (a partir de qué número de la matriz de dispersion va a reemplazar los valores en la imagen)
        print("Mediana de la imagen (valor por el que va a cambiar los pixeles seleccionados): ", moda)
        print("Valor recomendado: cercas de 70")
        filtro = int(input("Introduce el filtro que quieres colocar (0 - 100): "))

        # Reemplazar los valores por la moda
        if len(modas) > 1:
            # Si hay mas de una moda usamos un mulltimodal
            imagen_filtrada_multimoda(imagen_recortada)
        else:
            # Si solo es una moda procedemos normal
            imagen_filtrada(imagen_recortada)

        # Promedio de la imagen filtrada resultante
        promedio_final = np.mean(imagen_filtrada_con_moda)

        # Obtener la matriz de dispersion resultante
        matriz_de_dispersion_resultante = crear_matriz_dispersion(imagen_filtrada_con_moda, promedio_final)

        # Dispersion de la matriz resultante final
        print("El valor final de la sumatoria de la matriz resultante es la siguiente: ", np.sum(matriz_de_dispersion_resultante))

        np.savetxt('imagen_original.csv', imagen_recortada, delimiter=',', fmt='%d')
        np.savetxt('matriz_de_dispersion_moda.csv', matriz_de_dispersion, delimiter=',', fmt='%d') 
        np.savetxt('imagen_filtrada_con_moda.csv', imagen_filtrada_con_moda, delimiter=',', fmt='%d')
        np.savetxt('matriz_de_dispersion_moda_resultante.csv', matriz_de_dispersion_resultante, delimiter=',', fmt='%d')

        # Mostrar la imagen recortada
        cv2.imshow("Imagen original", imagen_recortada)
        cv2.imshow("matriz de dispersion moda", matriz_de_dispersion)
        cv2.imshow("imagen filtrada con moda", imagen_filtrada_con_moda)
        cv2.imshow("Matriz de dispersion resultante", matriz_de_dispersion_resultante)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Guardar la imagen en los archivos
        cv2.imwrite("imagen_filtrada_con_moda.png", imagen_filtrada_con_moda)