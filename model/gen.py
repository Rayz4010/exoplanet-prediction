from PIL import Image
import numpy as np

size = 16
palette = np.array([
    (25,45,70), 
    (80,110,160), 
    (50,200,140), 
    (240,220,30), 
    (100,210,100), 
    (170,60,190)
], dtype=np.uint8)

indices = np.random.randint(0, len(palette), (size, size))
img_array = palette[indices]

img = Image.fromarray(img_array, "RGB").resize((256, 256), Image.NEAREST)

path = "blob_v2.png"
img.save(path)

path
