from PIL import Image
import requests
from io import BytesIO
import torch 

# Image url
image_url = "https://jemalawyers.com.pg/wp-content/uploads/2023/02/Lawyers-Group-Photo-2-scaled.jpg"

# Load Image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image.show()


model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
# Inference
results = model(image)
# Results
results.print()
# Show detection
results.show()