import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Oxford-IIIT Pet dataset
DATA_DIR = "./data"

train_dataset = OxfordIIITPet(
    root=DATA_DIR,
    split='trainval',
    target_types='category',
    download=True,
    transform=train_transform
)

test_dataset = OxfordIIITPet(
    root=DATA_DIR,
    split='test',
    target_types='category',
    download=True,
    transform=test_transform
)

total = len(train_dataset)
val_size = int(0.2*total)
train_size = total - val_size

train_set, val_set = random_split(train_dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)

val_loader = DataLoader(
    val_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)


# Initialize model
model = resnet34(weights=ResNet34_Weights.DEFAULT)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)

for param in model.layer1.parameters():
    param.requires_grad = True
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True


model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-4},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),    'lr': 1e-4},
])

# Replace final layer

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# Training loop
best_acc = 0.0
epochs_without_improvement = 0
patience = 5
max_epochs = 20
for epoch in range(1, max_epochs + 1):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} - Training"):  
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Evaluation phase
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} - Testing"):  
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
    epoch_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Validation Accuracy = {epoch_acc:.4f}")

    # Save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'strat1results.pth')
        print(f'PB! {best_acc:.4f}')
    else:
        epochs_without_improvement += 1
        print(f'No improvement for {epochs_without_improvement}/{patience} epochs')

    if epochs_without_improvement >= patience:
        print(f'Stopping early due to no patince: within last {patience} epochs')
        break
model.eval()
correct = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc=f"Final test"):  
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
test_accuracy = correct / len(test_loader.dataset)

print(f"Best Test Accuracy Achieved: {test_accuracy:.4f}")
