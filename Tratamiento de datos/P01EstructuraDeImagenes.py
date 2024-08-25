import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread("Incendio.png")
if imagen is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo 'Incendio.png' está en el directorio correcto.")
else:
    # Divide la imagen en sus canales B, G y R
    b, g, r = cv2.split(imagen)

    # Guardar cada canal en archivos CSV separados
    np.savetxt('canal_azul.csv', b, delimiter=',', fmt='%d')
    np.savetxt('canal_verde.csv', g, delimiter=',', fmt='%d')
    np.savetxt('canal_rojo.csv', r, delimiter=',', fmt='%d')
    print("Los canales de color se han guardado en 'canal_azul.csv', 'canal_verde.csv', y 'canal_rojo.csv'.")

    # Mostrar la imagen original
    cv2.imshow("Imagen a Color", imagen)
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('imagen_gris.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_gris.csv'.")

    # Mostrar la imagen en blanco y negro
    cv2.imshow("Imagen a Blanco y negro", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
