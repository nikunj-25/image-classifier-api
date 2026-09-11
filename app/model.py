import torch
from model.train import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load('model/classifier.pth'))
model.eval()

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]