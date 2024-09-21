import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("Error: No se pudo cargar la imagen")
else:
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

        # Promedio de la imagen recortada
        promedio = np.mean(imagen_recortada)

        # mediana de la imagen recortada
        mediana = np.median(imagen_recortada)

        # Formula para la matriz de dispersion
        for i in range(imagen_recortada.shape[0]):
            for j in range(imagen_recortada.shape[1]):
                matriz_de_dispersion[i][j] = abs(imagen_recortada[i , j] - promedio)*100/promedio
        
        # Calcular el histograma de la imagen recortada
        histograma = cv2.calcHist([matriz_de_dispersion], [0], None, [256], [0, 256])

        # Crear un histograma de barras con matplotlib
        plt.figure()
        plt.title("Histograma de Barras - Matriz de dispersion")
        plt.xlabel("Valor de Intensidad (0-255)")
        plt.ylabel("Número de Píxeles")
        
        # Convertir el histograma en una lista y graficar usando barras
        plt.bar(range(256), histograma[:, 0], width=1, color='gray')
        plt.xlim([0, 255])
        plt.show()

        # Promedio de la imagen filtrada (matriz de dispersion)
        promedio_filtrada = np.mean(matriz_de_dispersion)

        # Filtro que se le va a colocar a la imagen (a partir de que numero de la matriz de dispersion va a reemplazar los valores en la imagen)
        print("Promedio de dispersion: ", promedio_filtrada)
        print("Mediana del recorte (valor por el que va a cambiar): ",mediana)
        filtro = int(input("Introduce el filtro que quieres colocar (0 - 255): "))

        for i in range(imagen_recortada.shape[0]):
            for j in range(imagen_recortada.shape[1]):
                if matriz_de_dispersion[i][j] >= filtro :
                    imagen_filtrada_con_mediana[i][j] = mediana

        #matriz_de_dispersion = cv2.medianBlur(imagen_recortada, 5)

        np.savetxt('imagen.csv', imagen, delimiter=',', fmt='%d')
        np.savetxt('imagen_recortada.csv', imagen_recortada, delimiter=',', fmt='%d')
        np.savetxt('imagen_filtrada_con_mediana.csv', imagen_filtrada_con_mediana, delimiter=',', fmt='%d')
        np.savetxt('matriz_de_dispersion.csv', matriz_de_dispersion, delimiter=',', fmt='%d') 

        # Mostrar la imagen recortada
        cv2.imshow("matriz_de_dispersion", matriz_de_dispersion)
        cv2.imshow("imagen_filtrada_con_mediana", imagen_filtrada_con_mediana)
        cv2.imshow("Imagen_recortada", imagen_recortada)
        #cv2.imshow("Imagen original", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
