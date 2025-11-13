import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define your model architecture ---
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Load your trained model ---
model = Net()
model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
model.eval()

# --- 3. Prepare the CIFAR-10 test dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
classes = testset.classes  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- 4. Pick a random image ---
idx = random.randint(0, len(testset) - 1)
image, label = testset[idx]

# --- 5. Run inference ---
with torch.no_grad():
    outputs = model(image.unsqueeze(0))
    _, predicted = torch.max(outputs, 1)
    predicted_label = classes[predicted.item()]
    true_label = classes[label]

# --- 6. Show and save the image ---
img_to_show = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
plt.imshow(img_to_show)
plt.title(f"True: {true_label} | Predicted: {predicted_label}")
plt.axis("off")
plt.show()

# (Optional) Save to file
plt.imsave("cifar10_prediction.png", img_to_show)
print(f"Image saved as 'cifar10_prediction.png'")
print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
