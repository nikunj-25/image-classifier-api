import torch
from model.train import SimpleCNN, transform
import torchvision

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

model = SimpleCNN()
model.load_state_dict(torch.load('model/classifier.pth'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")