from PIL import Image
import glob
from resizeimage import resizeimage
import re
import os

for img in glob.glob("/home/terry/Documents/ML/enclave_employee_images_orig/" + "*.JPG"):  # All jpeg images
    with open(img, 'r+b') as f:
        with Image.open(f) as image:
            print(img)
            cover = resizeimage.resize_cover(image, [480, 720])
            matchObj = re.search(r'\(([^)]*)\)', os.path.splitext(os.path.basename(img))[0], re.M | re.I)
            name = matchObj.group(1)
            cover.save('/home/terry/Documents/ML/enclave-employee-images/' + name + ".jpg", image.format)
