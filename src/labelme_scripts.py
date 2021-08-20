import labelme
import base64

img_path = '0_00001.jpg'
data = labelme.LabelFile.load_image_file(img_path)
image_data = base64.b64encode(data).decode('utf-8')
print("\"imageData\": \"" + image_data + "\",")
