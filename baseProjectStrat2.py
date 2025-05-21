import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import random_split
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

device = torch.device("cuda")



# Data transforms
train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256),
    v2.RandomResizedCrop(224, scale=(0.2, 1), ratio=(3/4,4/3)),
    v2.RandomHorizontalFlip(p=0.5),
    #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #v2.RandomRotation(degrees=15),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((256)),
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

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad   = False
        m.train()

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)
model = model.to(device)

for parameter in model.fc.parameters():
    parameter.requires_grad = True

unfreeze_schedule = {
    5: ['layer4'],
    10: ['layer3'],
    15: ['layer2'],
    20: ['layer1']

}

    def unfreeze_layer(model, names):
        for name, module in model.named_children():
            if name in names:
                for p in module.parameters():
                    p.requires_grad = True



    # Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max = 20, eta_min=1e-6)
# Training loop
best_acc = 0.0
epochs_without_improvement = 0
patience = 5
max_epochs = 20
eta = 1e-4

for epoch in range(1, max_epochs + 1):

    if epoch in unfreeze_schedule:
        next_layer = unfreeze_schedule[epoch]
        print(f'Unfreezing {next_layer} at epoch {epoch}')
        unfreeze_layer(model, next_layer)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=eta, weight_decay=1e-3)

        remaining = max_epochs -epoch+1
        scheduler = CosineAnnealingLR(optimizer, remaining,1e-6)
        eta *= 0.1
        if eta < 1e-6:
            eta = 1e-6


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

    scheduler.step()

    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Validation Accuracy = {epoch_acc:.4f}")

    # Save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'bestmodel.pth')
        print(f'PB! {best_acc:.4f}')
    else:
        epochs_without_improvement += 1
        print(f'No improvement for {epochs_without_improvement}/{patience} epochs')

    if epochs_without_improvement >= patience:
        print(f'Stopping early due to no patince: within last {patience} epochs')
        break


model.load_state_dict(torch.load('bestmodel.pth'))
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
