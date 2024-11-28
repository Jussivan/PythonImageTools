import cv2
import numpy as np
from google.colab.patches import cv2_imshow

imagem = cv2.imread("sua_imagem.png")
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
niveis = int(input("Informe a quantidade de cores desejadas para a imagem processada: "))
intervalo = 256 // niveis
imagem_quantizada = (imagem_cinza // intervalo) * intervalo
cv2.imwrite("imagem_quantizada.jpg", imagem_quantizada)
cv2_imshow(imagem_quantizada)