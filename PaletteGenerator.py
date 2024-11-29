import cv2
import numpy as np
from sklearn.cluster import KMeans

image = cv2.imread("sua_imagem.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (100, 100))

pixels = image.reshape(-1, 3)

niveis = int(input("Digite o número de cores desejados para sua paleta: "))

kmeans = KMeans(n_clusters=niveis)
kmeans.fit(pixels)

quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
quantized_image = quantized_pixels.reshape(image.shape).astype(np.uint8)

kmeans = cv2.kmeans(pixels.astype(np.float32), n_colors, None, 
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                    10, cv2.KMEANS_RANDOM_CENTERS)[2]
palette = kmeans.astype(int)

cv2.imwrite("quantized_image.png", cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(8, 4))
plt.title('Paleta de Cores')
plt.axis('off')
for i, color in enumerate(palette):
    plt.fill_between([i, i + 1], 0, 1, color=color / 255)
plt.xlim(0, len(palette))
plt.show()