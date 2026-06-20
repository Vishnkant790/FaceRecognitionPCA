import os
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = "data/raw/dataset/faces"

person_name = "Aamir"
person_folder = os.path.join(DATASET_PATH, person_name)

print(person_folder)
person_images = os.listdir(person_folder)
print(person_images)

image_name = person_images[0]
image_path = os.path.join(person_folder, image_name)
print(image_name)
print(image_path)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(type(image))
print(image.shape)

plt.imshow(image, cmap='gray')
plt.title(person_name)
plt.axis('off')
plt.show()