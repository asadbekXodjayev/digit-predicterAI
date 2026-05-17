import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Define a CNN (Convolutional Neural Network) for better image recognition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully connected layers (64 * 7 * 7 = 3136 after 2 pooling operations)
        # Input: 28x28 → after pool1: 14x14 → after pool2: 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # First conv block (28x28 → 14x14)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block (14x14 → 7x7)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and fully connected (7x7 * 64 = 3136)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout3(nn.functional.relu(self.bn5(self.fc1(x))))
        x = self.fc2(x)
        
        return x

def train_model(epochs=15, batch_size=128, learning_rate=0.001):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transform for MNIST dataset with data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST training dataset
    print("Downloading MNIST training dataset...")
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Download and load MNIST test dataset
    print("Downloading MNIST test dataset...")
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    # Initialize model, loss function, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print("\nStarting training...")
    best_accuracy = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        scheduler.step()
        
        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    
    # Evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class statistics
            for i in range(len(target)):
                label = target[i].item()
                if predicted[i].item() == label:
                    class_correct[label] += 1
                class_total[label] += 1
    
    test_accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  Digit {i}: {acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print(f"\nModel saved to mnist_model.pth")
    
    return model, test_accuracy

if __name__ == "__main__":
    train_model(epochs=15, batch_size=128, learning_rate=0.001)