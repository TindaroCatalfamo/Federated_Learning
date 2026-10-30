import os
import sys
import csv
import json
import copy
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Assicura la corretta risoluzione dei moduli sia dalla root che da sottocartelle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from get_data import get_fashion_mnist_data
from models import MNISTCNN
from server import fed_avg

# Mappatura delle 10 classi del Fashion-MNIST
FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# ==============================================================================
# 1. EARLY STOPPING (Rigidita identica a FedDF Calibrated)
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
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
            print(f"[Early Stopping] Nessun miglioramento: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True


# ==============================================================================
# 2. ADDESTRAMENTO LOCALE DEI CLIENT
# ==============================================================================
def train_client_local(model, train_loader, client_id, epochs=20, lr=0.01, device='cpu'):
    """
    Esegue l'addestramento locale del client k per un numero specificato di epoche.
    Restituisce il modello aggiornato, il dizionario dei pesi, la loss e l'accuracy finale.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    last_avg_loss = 0.0
    last_accuracy = 0.0

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

            epoch_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        last_avg_loss = epoch_loss / total if total > 0 else 0.0
        last_accuracy = 100.0 * correct / total if total > 0 else 0.0

    return model, copy.deepcopy(model.state_dict()), last_avg_loss, last_accuracy


# ==============================================================================
# 3. VALUTAZIONI METRICHE GLOBALI (Validation e Test per classe)
# ==============================================================================
def evaluate_model_full(model, data_loader, device='cpu'):
    """
    Valuta il modello globale calcolando:
    - Loss media (CrossEntropy)
    - Accuratezza globale (%)
    - Accuratezza per classe (%) su tutte le 10 classi di Fashion-MNIST
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    num_classes = len(FASHION_CLASSES)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                lbl = labels[i].item()
                class_correct[lbl] += (predicted[i] == lbl).item()
                class_total[lbl] += 1

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[FASHION_CLASSES[i]] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_acc[FASHION_CLASSES[i]] = 0.0

    return avg_loss, accuracy, class_acc


# ==============================================================================
# 4. SALVATAGGIO GRAFICI E STORICO CSV
# ==============================================================================
def save_plots_and_history(history, cartella_destinazione, scenario_name):
    """
    Esporta il file CSV completo e genera i grafici ad alta risoluzione delle curve di apprendimento.
    """
    os.makedirs(cartella_destinazione, exist_ok=True)
    rounds = range(1, len(history['round']) + 1)

    # 1. Scrittura CSV
    csv_path = os.path.join(cartella_destinazione, 'global_history.csv')
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Round',
            'Val_Loss', 'Val_Accuracy',
            'Test_Loss', 'Test_Accuracy'
        ] + FASHION_CLASSES
        writer.writerow(header)

        for i, r in enumerate(rounds):
            row = [
                r,
                f"{history['val_loss'][i]:.4f}",
                f"{history['val_accuracy'][i]:.2f}",
                f"{history['test_loss'][i]:.4f}",
                f"{history['test_accuracy'][i]:.2f}"
            ]
            row.extend([f"{history['test_class_acc'][i][cls]:.2f}" for cls in FASHION_CLASSES])
            writer.writerow(row)

    print(f"\n[OK] Storico numerico FedAvg salvato in: {csv_path}")

    # 2. Generazione grafici
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Grafico 1: Loss
    ax1.plot(rounds, history['val_loss'], label='Server Val Loss (10%)', color='#e67e22', marker='o', markersize=3, linewidth=2)
    ax1.plot(rounds, history['test_loss'], label='Server Test Loss', color='#c0392b', marker='s', markersize=3, linewidth=1.5, alpha=0.8)
    ax1.set_title('Loss Curves (FedAvg)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Grafico 2: Accuracy
    ax2.plot(rounds, history['val_accuracy'], label='Server Val Accuracy', color='#27ae60', marker='o', markersize=3, linewidth=2)
    ax2.plot(rounds, history['test_accuracy'], label='Server Test Accuracy', color='#2980b9', marker='s', markersize=3, linewidth=2)
    ax2.set_title('Accuracy Curves (FedAvg)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # Grafico 3: Per-Class Accuracy
    cmap = plt.get_cmap('tab10')
    for idx, cls in enumerate(FASHION_CLASSES):
        cls_vals = [history['test_class_acc'][r_idx][cls] for r_idx in range(len(rounds))]
        ax3.plot(rounds, cls_vals, label=cls, color=cmap(idx), linewidth=1.4, alpha=0.85)
    ax3.set_title('Test Accuracy per Classe (FedAvg)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=8, ncol=2, loc='lower right')

    plt.suptitle(f'FedAvg Baseline: {scenario_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(cartella_destinazione, 'learning_curves.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafici salvati in: {plot_path}")


# ==============================================================================
# 5. PIPELINE PRINCIPALE FEDAVG
# ==============================================================================
def run_fedavg(
    seed=42,
    alpha=0.1,
    num_clients=5,
    local_epochs=20,
    client_lr=0.01,
    patience=20,
    num_rounds=100,
    batch_size=64,
    output_dir=None,
    exp_name='00_FedAvg'
):
    start_time = time.time()

    # Se output_dir non è specificato, usa automaticamente la cartella del seed corrispondente
    if output_dir is None:
        dest_folder = os.path.join('./esperimenti_finali_FedDF', f'seed{seed}', exp_name)
    else:
        dest_folder = os.path.join(output_dir, exp_name)

    os.makedirs(dest_folder, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 75)
    print(f"=== AVVIO BASELINE FEDAVG [Seed: {seed}] ===")
    print("=" * 75)
    print(f" - Cartella output:        {dest_folder}")
    print(f" - Dirichlet Alpha:        {alpha} (Eterogeneita Non-IID)")
    print(f" - Numero Client:          {num_clients}")
    print(f" - Epoche Locali Client:   {local_epochs}")
    print(f" - Client Learning Rate:   {client_lr} (SGD)")
    print(f" - Seed:                   {seed}")
    print(f" - Pazienza Early Stop:    {patience} round su Server Validation Loss")
    print(f" - Dispositivo:            {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    # 1. Caricamento Dataset e Partizioni deterministiche con lo STESSO seed di FedDF
    client_loaders, _, val_loader, test_loader = get_fashion_mnist_data(
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size,
        val_size=6000,
        num_distill_samples=10000,
        seed=seed
    )

    client_sizes = [len(client_loaders[i].dataset) for i in range(num_clients)]

    # 2. Inizializzazione Modello Globale
    torch.manual_seed(seed)
    global_model = MNISTCNN()
    global_weights = copy.deepcopy(global_model.state_dict())
    early_stopping = EarlyStopping(patience=patience)

    history = {
        'round': [],
        'val_loss': [],
        'val_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_class_acc': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_round = 0
    test_acc_at_best_val = 0.0

    # === CICLO DEI ROUND DI COMUNICAZIONE FEDAVG ===
    for round_idx in range(1, num_rounds + 1):
        round_t0 = time.time()
        print(f"\n{'-'*75}\n[FEDAVG ROUND {round_idx:02d}/{num_rounds}] (Seed {seed}) In corso...\n{'-'*75}")

        # 1. Addestramento Locale Client
        client_weights_dict = {}
        for i in range(num_clients):
            c_model = MNISTCNN()
            c_model.load_state_dict(copy.deepcopy(global_weights))

            c_model, weights, c_loss, c_acc = train_client_local(
                model=c_model,
                train_loader=client_loaders[i],
                client_id=i+1,
                epochs=local_epochs,
                lr=client_lr,
                device=DEVICE
            )
            client_weights_dict[f"client_{i+1}"] = weights
            print(f"  -> Client {i+1} addestrato ({local_epochs} ep.) | Train Loss: {c_loss:.4f} | Train Acc: {c_acc:.2f}%")

        # 2. Aggregazione dei Pesi via FedAvg Ponderato (n_k / N)
        global_weights = fed_avg(client_weights_dict, client_sizes=client_sizes)
        global_model.load_state_dict(global_weights)

        # 3. Valutazione sul Server Validation Set (per Early Stopping e Model Selection)
        stud_v_loss, stud_v_acc, _ = evaluate_model_full(global_model, val_loader, device=DEVICE)

        # 4. Valutazione sul Test Set Globale
        stud_t_loss, stud_t_acc, stud_cls_acc = evaluate_model_full(global_model, test_loader, device=DEVICE)

        round_duration = time.time() - round_t0

        # Registrazione Storico
        history['round'].append(round_idx)
        history['val_loss'].append(stud_v_loss)
        history['val_accuracy'].append(stud_v_acc)
        history['test_loss'].append(stud_t_loss)
        history['test_accuracy'].append(stud_t_acc)
        history['test_class_acc'].append(stud_cls_acc)

        print(f"\n[SERVER FEDAVG ROUND {round_idx}] ({round_duration:.1f}s)")
        print(f"  Server Val Accuracy:   {stud_v_acc:.2f}% (Loss: {stud_v_loss:.4f})")
        print(f"  Server Test Accuracy:  {stud_t_acc:.2f}% (Loss: {stud_t_loss:.4f})")

        # Checkpoint ed Early Stopping (guidato da Server Validation Loss)
        is_best = early_stopping(stud_v_loss)
        torch.save(global_weights, os.path.join(dest_folder, 'globale_post_fed_ultimo_round.pt'))

        if is_best:
            best_val_loss = stud_v_loss
            best_val_acc = stud_v_acc
            best_round = round_idx
            test_acc_at_best_val = stud_t_acc
            torch.save(global_weights, os.path.join(dest_folder, 'miglior_modello_globale.pt'))
            print(f"  --> [CHECKPOINT] Nuovo Miglior Modello FedAvg Salvato! (Best Val Loss: {best_val_loss:.4f})")

        if early_stopping.early_stop:
            print(f"\n[!] EARLY STOPPING attivato al Round {round_idx} (Nessun miglioramento su Val Loss per {patience} round).")
            break

    total_training_time = time.time() - start_time
    print(f"\n=== Addestramento FedAvg completato in {total_training_time/60:.1f} minuti ===")

    # Salvataggio CSV e Grafici
    save_plots_and_history(history, dest_folder, f"FedAvg_seed{seed}")

    # Salvataggio Summary JSON (compatibile al 100% con i formati di FedDF)
    summary_data = {
        'experiment_name': exp_name,
        'method': 'FedAvg',
        'alpha': alpha,
        'num_clients': num_clients,
        'local_epochs': local_epochs,
        'client_lr': client_lr,
        'patience': patience,
        'seed': seed,
        'device': str(DEVICE),
        'total_rounds_executed': len(history['round']),
        'best_round': best_round,
        'best_student_val_loss': best_val_loss,
        'best_student_val_acc': best_val_acc,
        'student_test_acc_at_best_val': test_acc_at_best_val,
        'final_student_test_acc': history['test_accuracy'][-1],
        'total_duration_minutes': round(total_training_time / 60.0, 2)
    }

    summary_path = os.path.join(dest_folder, 'summary_esperimento.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"[OK] Summary JSON FedAvg salvato in: {summary_path}")

    print("\n" + "=" * 75)
    print(f"=== RISULTATO FINALE FEDAVG [Seed {seed}] ===")
    print(f" - Miglior Round (Val Loss):      {best_round}")
    print(f" - Best Val Accuracy:             {best_val_acc:.2f}% (Loss: {best_val_loss:.4f})")
    print(f" - Test Accuracy al Best Val:     {test_acc_at_best_val:.2f}%")
    print(f" - Test Accuracy Ultimo Round:    {history['test_accuracy'][-1]:.2f}%")
    print("=" * 75 + "\n")

    return summary_data


# ==============================================================================
# 6. PARSER DA RIGA DI COMANDO
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg Baseline Multi-Seed (Fashion-MNIST)")

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per la riproducibilita e partizionamento (es. 42, 43, 44)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Parametro alpha della distribuzione di Dirichlet')
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Numero di client federati')
    parser.add_argument('--local_epochs', type=int, default=20,
                        help='Epoche di addestramento locale per client')
    parser.add_argument('--client_lr', type=float, default=0.01,
                        help='Learning rate SGD locale dei client')
    parser.add_argument('--patience', type=int, default=20,
                        help='Pazienza per Early Stopping su Server Validation Loss')
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='Numero massimo di round di comunicazione')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Dimensione del batch')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Cartella base di output (default: ./esperimenti_finali_FedDF/seed{seed})')
    parser.add_argument('--exp_name', type=str, default='00_FedAvg',
                        help='Nome identificativo dello scenario')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run_fedavg(
        seed=args.seed,
        alpha=args.alpha,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        client_lr=args.client_lr,
        patience=args.patience,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        exp_name=args.exp_name
    )

