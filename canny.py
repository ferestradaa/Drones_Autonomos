import cv2
import numpy as np


def gaussian_fiter(gray): #funcion para aplicar filtro gaussiano para pocesamiento de imagen
    kernel_size = (5,5)
    sigma = 1.0
    gauss = cv2.GaussianBlur(gray, kernel_size, sigma)
    return gauss

def canny(gray): #funcion para aplicar deteccion de border con canny
    lower_threshold = 50 #asignacion de umbral para gradientes
    upper_threshold = 150
    canny_edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    return canny_edges


def dilate_filter(canny): #funcion para aplicar dilatacion y destacar los border
    kernel = np.ones((2, 2), np.uint8) #kernel para que no se vuelvan tan marcados los borders
    dilated_image = cv2.dilate(canny, kernel, iterations=1)
    return dilated_image
    
video = 'scenario.mp4' #lectura del video 
cap = cv2.VideoCapture(video)

while cap.isOpened(): #lecura de cada fotograma
    ret, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertir frame a escala de grises
    gauss_frame = gaussian_fiter(gray_frame) #aplicar filtro gausseano
    canny_frame = canny(gauss_frame) #aplicar canny
    filtered_image = dilate_filter(canny_frame) #aplicar filtro para ruido 
    cv2.imshow('Canny', filtered_image) #desplegar video con procesamiento
    
    # Presionar 'q' para salir
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
