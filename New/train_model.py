import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GymGRUModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GymGRUModel, self).__init__()
        
        self.gru1 = nn.GRU(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        self.gru2 = nn.GRU(64, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        
        out, _ = self.gru2(out)
        
        out = out[:, -1, :]
        
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

print("Model definition complete.")
print("Loading data...")

X = np.load('X_data.npy')
y = np.load('y_data.npy')
classes = np.load('classes.npy')

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {classes}")

y_indices = np.argmax(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y_indices, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data is ready for training.")

input_size = X.shape[2]
num_classes = y.shape[1]

model = GymGRUModel(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 150
patience = 15
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = None

print("Model and optimizer initialized and ready.")
print("Starting GRU model training...")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
        
    epoch_train_loss = train_loss / len(train_loader.dataset)
    epoch_train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    epoch_val_loss = val_loss / len(test_loader.dataset)
    epoch_val_acc = val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{epochs}] | '
          f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_weights = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}.')
            break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

model_path = 'gym_gru_model.pth'
torch.save(model.state_dict(), model_path)

print(f"Model (best weights) saved successfully at: {model_path}")