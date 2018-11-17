!pip install imageio
from skimage import img_as_float
import imageio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

image_filename = 'image1.jpg'
img = imageio.imread(image_filename)
print(img.shape)
plt.axis('off') 
plt.imshow(img)
print('\n')

#Okay, so the array has 711 rows each of pixel 996x3. Let's reshape it into a format that PCA can understand.
# 2988 = 996 * 3
img_r = np.reshape(img, (img.shape[0], img.shape[1]*3))
print(img_r.shape)

ipca = PCA(64).fit(img_r) 
img_c = ipca.transform(img_r) 
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))
plt.imshow(img)

#OK, now to visualize how PCA has performed this compression, let's inverse transform the PCA output and #reshape for visualization using imshow. 
temp = ipca.inverse_transform(img_c) 
print(temp.shape)
#reshaping 2988 back to the original 996 * 3 
# temp = img_as_float(temp)
# temp = np.mean(temp)
temp = np.reshape(temp, (img.shape[0], img.shape[1])) 
# temp = np.random.uniform(0, 200000000, (1, img.shape[0], img.shape[1], 3))
print(temp.shape)

# plt.axis('off')
plt.imshow(temp)
