import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import random
from collections import defaultdict

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
    transform=train_transform,
    target_transform= lambda y: torch.tensor(y)
)

test_dataset = OxfordIIITPet(
    root=DATA_DIR,
    split='test',
    target_types='category',
    download=True,
    transform=test_transform,
    target_transform= lambda y: torch.tensor(y)
)

# Split training into Train & Val sets
total = len(train_dataset)
val_size = int(0.2*total)
train_size = total - val_size

train_set, val_set = random_split(train_dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))


#COMMENT OUT IF WANT FULL TRAIN, VAL AND TEST SETS
#small_train_size = 2000
#small_val_size = 250
#small_test_size = 500
#train_set, _ = random_split(train_set, [small_train_size, len(train_set) - small_train_size], generator=torch.Generator().manual_seed(42))
#val_set, _ = random_split(val_set, [small_val_size, len(val_set) - small_val_size], generator=torch.Generator().manual_seed(42))
#test_dataset, _ = random_split(test_dataset, [small_test_size, len(test_dataset) - small_test_size], generator=torch.Generator().manual_seed(42))

# Split trainingset into labeled and unlabeled
initial_labeled_ratio = 0.10
data_size = len(train_set)
labeled_data_size = int(initial_labeled_ratio * data_size)

# Get inidces for labeled and unlabeled dataset
indices = list(range(data_size))
random.shuffle(indices)
labeled_indices = indices[:labeled_data_size]
unlabeled_indices = indices[labeled_data_size:]

labeled_dataset = Subset(train_set, labeled_indices)
unlabeled_dataset = Subset(train_set, unlabeled_indices)

# Loaders: Labeled, Unlabled, Val, Test
labeled_loader = DataLoader(
    labeled_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
unlabeled_loader = DataLoader(
    unlabeled_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)



# Self training hyperparameters
SELF_TRAINING = True
pseudo_threshold = 0.90
num_self_training_epochs = 5
new_labels_counter = []   # To plot in results
pseudo_labels_records = [] # List of dictionaries: One per self_trianing epoch
list_of_label_accuracies = []

for self_training_epoch in range(1, num_self_training_epochs + 1):
    print(f"Self-training epoch {self_training_epoch}")

    # Initialize ResNet34
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = True
            m.bias.requires_grad = True

    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)
    model = model.to(device)

    for parameter in model.fc.parameters(): # start with unfreezing head
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

    # Loss and optimizer (only fine-tuning final layer)
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
        for inputs, labels in tqdm(labeled_loader, desc=f"Epoch {epoch}/{max_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(labeled_loader.dataset)

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
            torch.save(model.state_dict(), 'unfreezingRes2.pth')
            print(f'PB! {best_acc:.4f}')
        else:
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement}/{patience} epochs')

        if epochs_without_improvement >= patience:
            print(f'Stopping early due to no patince: within last {patience} epochs')
            break

    model.load_state_dict(torch.load('unfreezingRes2.pth'))   # Load best model from training
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


    if not SELF_TRAINING:
        break
    # Pseudo labeling
    model.load_state_dict(torch.load('unfreezingRes2.pth'))   # Load best model from training
    model.eval()
    epoch_labels_record = defaultdict(list)
    indices_to_remove = []
    all_pseudo_inputs = []
    all_pseudo_labels = []
    total_new_labels = 0
    total_new_correct_labels = 0
    with torch.no_grad():
        for i, (inputs, true_labels) in enumerate(tqdm(unlabeled_loader, desc=f"Self-training Epoch {self_training_epoch}/{num_self_training_epochs} - Pseudo-labeling")):
            batch_start = i * unlabeled_loader.batch_size
            batch_end = batch_start + inputs.size(0)
            current_indices = unlabeled_indices[batch_start:batch_end]

            inputs = inputs.to(device)
            outputs = model(inputs)
            probs, preds = torch.max(torch.softmax(outputs, dim=1), 1)

            new_labels = probs >= pseudo_threshold
            new_labels_batch = new_labels.sum().item()
            if new_labels_batch > 0:
                total_new_labels += new_labels_batch

                # Add new datapoints with predicted pseudo label from batch
                all_pseudo_inputs.append(inputs[new_labels].cpu())
                all_pseudo_labels.append(preds[new_labels].cpu())

                # Find indices of data to remove from unlabeled dataset
                new_indices_to_remove = []
                for j in range(len(new_labels)):
                    if new_labels[j]:
                        # Save True_label : Predcited_label
                        true_label = true_labels[j].item()
                        pred_label = preds[j].item()
                        epoch_labels_record[true_label].append(pred_label)
                        if true_label == pred_label:
                            total_new_correct_labels += 1
                        new_indices_to_remove.append(current_indices[j])
                indices_to_remove.extend(new_indices_to_remove)
                #print(indices_to_remove)

                #print(f"Added {new_labels_batch} new samples to labeled dataset")

    if total_new_labels == 0:
        print("No new labels found: Stopping self-trainig early")
        print(pseudo_labels_records)
        print("List of label_acuraccies per self-training epoch: ")
        print(list_of_label_accuracies)
        break
    else:
        # Add new pseudo-labeled datapoints to training set
        pseudo_inputs = torch.cat(all_pseudo_inputs, dim=0)
        pseudo_labels = torch.cat(all_pseudo_labels, dim=0)
        new_labeled_subset = TensorDataset(pseudo_inputs, pseudo_labels)
        labeled_dataset = ConcatDataset([labeled_dataset, new_labeled_subset])
        labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

        # Remove from unlabeled
        unlabeled_indices = [i for i in unlabeled_indices if i not in indices_to_remove]
        unlabeled_dataset = Subset(train_set, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        # Save number of new labels for this self trainnig epoch
        new_labels_counter.append(total_new_labels)
        print(f"Added {total_new_labels} new samples to labeled dataset")

        # Store label predictions for this self training epoch
        pseudo_labels_records.append(epoch_labels_record)
        
        # Print accuracy of newly added labels
        print(f"Accuracy of newly added labels: {total_new_correct_labels / total_new_labels}")
        list_of_label_accuracies.append(total_new_correct_labels / total_new_labels)

    print(pseudo_labels_records)
    print("List of label_acuraccies per self-training epoch: ")
    print(list_of_label_accuracies)
