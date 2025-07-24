import torch
from tqdm import tqdm
from torch import nn, optim
from model import get_resnet18
from data_loader import get_data_loaders
from utils import save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataloaders, num_epochs=10, lr=0.001, patience=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    early_stop_counter = 0
    loss_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloaders['train'], desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloaders['train'])
        loss_history.append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f'Val Accuracy: {val_acc:.4f}')

        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0
            save_model(model, f'resnet18_best.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    return model, loss_history


