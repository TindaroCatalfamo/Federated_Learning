import torch
import torch.nn as nn
import torch.optim as optim

def train_client(model, train_loader, client_id, epochs=5, lr=0.01, device='cpu'):
    print(f"--- Avvio Addestramento Client {client_id} ---")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoca {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    return model, model.state_dict(), history