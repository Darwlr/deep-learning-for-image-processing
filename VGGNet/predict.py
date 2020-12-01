import torch
from model import vgg
import matplotlib.pyplot as plt
from torchvision import transforms
import json
from PIL import Image

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img = Image.open("../predict_image/rose.jpg")
plt.imshow(img)

img = data_transform(img) # [C, H, W]
img = torch.unsqueeze(img, dim=0) # [N, C, H, W]

# read class_dict
try:
    json_file = open('class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = vgg(model_name="vgg16", num_classes=5)
# load_weights
model_weight_path = 'vgg16Net.pth'
model.load_state_dict(torch.load(model_weight_path))
model.eval()

with torch.no_grad():
    output = model(img)
    predict_y = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict_y).numpy()
print(class_indict[str(predict_cla)])
plt.show()
