import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


DATA_DIR = 'dataset'
BATCH_SIZE = 4        
EPOCHS = 50           
LEARNING_RATE = 0.0001 

class StegoNet(nn.Module):
    def __init__(self):
        super(StegoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.round(x * 255.0) % 2.0
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StegoNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {accuracy:.2f}%")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(BASE_DIR, 'stego_model.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete! Model successfully saved to: {save_path}")
