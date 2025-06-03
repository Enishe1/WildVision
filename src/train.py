import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import time
import copy


def create_directory_structure():
    """Create necessary directories for outputs"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)


def prepare_data(data_dir='data/images', val_split=0.2):
    """Prepare data loaders with augmentation and automatic train/val split"""
    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

    # Split into train and val
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transform to validation set
    val_dataset.dataset.transform = data_transforms['val']

    # Create dataloaders
    batch_size = 32
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    # Save class indices
    class_indices = full_dataset.class_to_idx
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    # Create image_datasets dict for compatibility
    image_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'classes': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx
    }

    return dataloaders, image_datasets


def create_model(num_classes):
    """Create ResNet50 model with custom classifier"""
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    return model


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """Train the model and return training history"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} phase'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'models/best_model.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def generate_evaluation_plots(history, model, dataloaders, class_names):
    """Generate evaluation visualizations"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()

    # Confusion matrix
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv('reports/classification_report.csv')

    # Per-class accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = (np.array(all_labels) == i)
        if sum(class_mask) > 0:  # Avoid division by zero
            class_acc = np.mean(np.array(all_preds)[class_mask] == i)
            class_accuracy[class_name] = class_acc

    plt.figure(figsize=(10, 25))
    sns.barplot(
        x=list(class_accuracy.values()),
        y=list(class_accuracy.keys()),
        orient='h'
    )
    plt.title('Per-Class Accuracy')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig('plots/per_class_accuracy.png')
    plt.close()


def train_and_evaluate():
    """Main training and evaluation pipeline"""
    # Setup environment
    create_directory_structure()

    # Prepare data
    dataloaders, image_datasets = prepare_data()
    num_classes = len(image_datasets['classes'])
    class_names = list(image_datasets['class_to_idx'].keys())

    # Create model
    model = create_model(num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=25
    )

    # Save final model
    torch.save(model.state_dict(), 'models/final_model.pth')

    # Generate evaluation plots
    generate_evaluation_plots(history, model, dataloaders, class_names)

    return history


if __name__ == '__main__':
    train_and_evaluate()