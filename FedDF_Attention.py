import os
import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import csv
import copy

from get_data import get_fashion_mnist_data
from models import MNISTCNN_Attention
from client import train_client
from server import fed_df_fusion_attention, evaluate_model

# Mappatura delle classi del Fashion MNIST per una lettura più chiara nel CSV
FASHION_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True 
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"[Early Stopping] Nessun miglioramento: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True

def evaluate_global_classwise(model, data_loader, num_classes=10, device='cpu'):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            # return_attention=False di default restituisce solo logits
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                
    avg_loss = test_loss / len(data_loader)
    overall_acc = 100.0 * correct / total
    
    class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[FASHION_CLASSES[i]] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_acc[FASHION_CLASSES[i]] = 0.0
            
    return avg_loss, overall_acc, class_acc

def plot_risultati_feddf_attention(history, round_effettivi, cartella_destinazione, beta):
    rounds = range(1, round_effettivi + 1)
    
    # 1. Salvataggio su CSV
    percorso_csv = os.path.join(cartella_destinazione, 'global_history_feddf_attention.csv')
    with open(percorso_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        intestazione = [
            'Round', 'Val_Loss', 'Val_Accuracy', 'Test_Loss', 'Test_Accuracy',
            'Distill_Total_Loss', 'Distill_KD_Loss', 'Distill_AT_Loss'
        ] + FASHION_CLASSES
        writer.writerow(intestazione)
        
        for r in range(round_effettivi):
            riga = [
                rounds[r], 
                history['val_loss'][r], 
                history['val_accuracy'][r],
                history['test_loss'][r], 
                history['test_accuracy'][r],
                history['distill_total_loss'][r],
                history['distill_kd_loss'][r],
                history['distill_at_loss'][r]
            ]
            riga.extend([history['test_class_acc'][r][cls] for cls in FASHION_CLASSES])
            writer.writerow(riga)
    print(f"Dati numerici salvati in: {percorso_csv}")
    
    # 2. Generazione Grafici a 3 pannelli
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Grafico 1: Loss
    ax1.plot(rounds, history['val_loss'], marker='o', markersize=3, color='orange', label='Server Val Loss (10%)', linewidth=2)
    ax1.plot(rounds, history['test_loss'], marker='s', markersize=3, color='red', label='Test Loss (Global)', linewidth=2)
    ax1.set_title('Loss per Round (FedDF + Attention)'), ax1.set_xlabel('Round'), ax1.grid(True, linestyle='--')
    ax1.legend()
    
    # Grafico 2: Accuracy
    ax2.plot(rounds, history['val_accuracy'], marker='o', markersize=3, color='cyan', label='Server Val Acc (10%)', linewidth=2)
    ax2.plot(rounds, history['test_accuracy'], marker='s', markersize=3, color='blue', label='Test Acc (Global)', linewidth=2)
    ax2.set_title('Accuracy per Round (FedDF + Attention)'), ax2.set_xlabel('Round'), ax2.grid(True, linestyle='--')
    ax2.legend()
    
    # Grafico 3: Distillation Losses
    ax3.plot(rounds, history['distill_total_loss'], marker='o', markersize=3, color='black', label='Total Distill Loss', linewidth=2)
    ax3.plot(rounds, history['distill_kd_loss'], marker='^', markersize=3, color='purple', label='KD Loss', linewidth=2)
    scaled_at = [at * beta for at in history['distill_at_loss']]
    ax3.plot(rounds, scaled_at, marker='d', markersize=3, color='green', label=f'AT Loss (beta={beta})', linewidth=2)
    ax3.set_title('Distillation Component Losses (Server)'), ax3.set_xlabel('Round'), ax3.grid(True, linestyle='--')
    ax3.legend()
    
    plt.savefig(os.path.join(cartella_destinazione, 'feddf_attention_learning_curves.png'), bbox_inches='tight')
    plt.close()

def main():
    print("=== Avvio Simulazione FL Multi-Round (FedDF + Activation-Based Attention Transfer) ===")
    
    # --- PANNELLO DI CONTROLLO ---
    NUM_CLIENTS = 5
    NUM_ROUNDS = 100
    EPOCHS_PER_ROUND = 5
    BATCH_SIZE = 64
    CLIENT_LR = 0.01
    
    # Iperparametri FedDF & Attentio
    # n Transfer
    DISTILLATION_EPOCHS = 1
    DISTILLATION_LR = 0.001
    TEMPERATURE = 2.0
    BETA = 10
    
    ALPHA = 0.1    
    PATIENCE = 20          
    SEED = 42

    scenario_name = f"FedDF_Attention_beta{int(BETA)}_alpha{ALPHA}_local_epochs{EPOCHS_PER_ROUND}_client{NUM_CLIENTS}_prova"
    cartella_pesi = f'./pesi_salvati/{scenario_name}'
    os.makedirs(cartella_pesi, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModalità: {scenario_name} | Client: {NUM_CLIENTS} | Device: {DEVICE} | Beta AT: {BETA} | Temp KD: {TEMPERATURE}")
    print(f"I risultati verranno salvati in: {cartella_pesi}\n")

    # Caricamento partizioni con dataset FAIR (stessi identici dati di FedAvg e FedDF standard)
    client_loaders, server_distill_loader, val_loader, test_loader = get_fashion_mnist_data(
        num_clients=NUM_CLIENTS, 
        alpha=ALPHA, 
        batch_size=BATCH_SIZE,
        val_size=6000,
        num_distill_samples=10000,
        seed=SEED
    )

    global_model = MNISTCNN_Attention()
    global_weights = global_model.state_dict()
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    global_history = {
        'val_loss': [], 
        'val_accuracy': [], 
        'test_loss': [], 
        'test_accuracy': [], 
        'test_class_acc': [],
        'distill_total_loss': [],
        'distill_kd_loss': [],
        'distill_at_loss': []
    }
    round_effettivi = 0
    
    # === CICLO DEI ROUND DI COMUNICAZIONE ===
    for round_idx in range(1, NUM_ROUNDS + 1):
        round_effettivi = round_idx
        print(f"\n{'='*40}\nROUND DI COMUNICAZIONE {round_idx}/{NUM_ROUNDS}\n{'='*40}")
        
        client_models_list = []
        
        # 1. Addestramento Locale dei Client
        for i in range(NUM_CLIENTS):
            client_model = MNISTCNN_Attention()
            client_model.load_state_dict(global_weights) 
            
            trained_model, weights, _ = train_client(
                model=client_model, 
                train_loader=client_loaders[i], 
                client_id=i+1, 
                epochs=EPOCHS_PER_ROUND, 
                lr=CLIENT_LR, 
                device=DEVICE
            )
            
            client_models_list.append(trained_model)    
            torch.save(weights, f'{cartella_pesi}/client_{i+1}_pre_fed.pt')
        
        # Dimensioni campioni per aggregazione ponderata
        client_sizes = [len(client_loaders[i].dataset) for i in range(NUM_CLIENTS)]
        
        # 2. Aggregazione FedDF + Attention Transfer sul Server
        print(f"\n[SERVER] Distillazione d'Ensemble FedDF + Attention Transfer (Round {round_idx})...")
        global_weights, avg_tot_loss, avg_kd_loss, avg_at_loss = fed_df_fusion_attention(
            global_model=global_model, 
            client_models=client_models_list, 
            unlabeled_loader=server_distill_loader, 
            client_sizes=client_sizes,
            epochs=DISTILLATION_EPOCHS, 
            lr=DISTILLATION_LR, 
            temperature=TEMPERATURE, 
            beta=BETA,
            device=DEVICE
        )
        global_model.load_state_dict(global_weights)
        
        # 3. Valutazione sul Server Validation Set (per Early Stopping)
        val_loss, val_acc = evaluate_model(global_model, val_loader, device=DEVICE)
        
        # 4. Valutazione sul Test Set Globale (metriche imparziali con accuratezza per classe)
        test_loss, test_acc, test_class_acc = evaluate_global_classwise(global_model, test_loader, device=DEVICE)
        
        # Registrazione metriche
        global_history['val_loss'].append(val_loss)
        global_history['val_accuracy'].append(val_acc)
        global_history['test_loss'].append(test_loss)
        global_history['test_accuracy'].append(test_acc)
        global_history['test_class_acc'].append(test_class_acc)
        global_history['distill_total_loss'].append(avg_tot_loss)
        global_history['distill_kd_loss'].append(avg_kd_loss)
        global_history['distill_at_loss'].append(avg_at_loss)
        
        print(f"[SERVER] Round {round_idx} -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Distill Loss: {avg_tot_loss:.4f} (KD: {avg_kd_loss:.4f}, AT: {avg_at_loss:.6f})")
        
        # --- CHECKPOINT & EARLY STOPPING SU VALIDATION SET ---
        is_best = early_stopping(val_loss)
        
        torch.save(global_weights, f'{cartella_pesi}/globale_post_fed_ultimo_round.pt')
        
        if is_best:
            print(f"--> [Checkpoint] Nuovo miglior modello globale registrato (Val Loss: {val_loss:.4f})")
            torch.save(global_weights, f'{cartella_pesi}/miglior_modello_globale.pt')
            
        if early_stopping.early_stop:
            print(f"\n[!] Early Stopping attivato al round {round_idx} (Nessun miglioramento su Val Loss per {PATIENCE} round)!")
            break
    
    # Salvataggio risultati e grafici completi
    plot_risultati_feddf_attention(global_history, round_effettivi, cartella_pesi, BETA)
    print(f"\n=== Simulazione FedDF + Attention Transfer Completata! I migliori pesi sono salvati in '{cartella_pesi}/miglior_modello_globale.pt' ===")

if __name__ == "__main__":
    main()
