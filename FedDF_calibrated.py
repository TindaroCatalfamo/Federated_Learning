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
import torch.nn.functional as F
import torch.optim as optim

# Assicura la corretta risoluzione dei moduli sia dalla root che dalla sottocartella
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from get_data import get_fashion_mnist_data
from models import MNISTCNN
from server import fed_avg

# Mappatura delle 10 classi del Fashion-MNIST
FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# ==============================================================================
# 1. EARLY STOPPING
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
# 3. VALUTAZIONI METRICHE (TEACHER ENSEMBLE E STUDENTE)
# ==============================================================================
def extract_client_class_counts(client_loaders, num_classes=10):
    """
    Estrae la matrice dei conteggi campioni per client e per classe: [num_clients, num_classes].
    """
    counts_list = []
    for loader in client_loaders:
        c_tensor = torch.zeros(num_classes, dtype=torch.float32)
        for _, labels in loader:
            for y in labels:
                c_tensor[y.item()] += 1.0
        counts_list.append(c_tensor)
    return torch.stack(counts_list)


def evaluate_teacher_ensemble(client_models, val_loader, tau=0.0, eps=None,
                              client_class_counts=None, device='cpu'):
    """
    Valuta l'Insegnante (Teacher Ensemble) sul Server Validation Set.
    Consente di monitorare le performance dell'ensemble pre-distillazione ad ogni round.
    """
    for model in client_models:
        model.to(device)
        model.eval()

    num_clients = len(client_models)
    num_classes = 10

    # Calcolo prior pi_k con Laplace smoothing se tau > 0
    client_priors_log = None
    if tau > 0.0 and client_class_counts is not None:
        smooth_counts = client_class_counts + 1.0
        pi = smooth_counts / smooth_counts.sum(dim=1, keepdim=True)
        client_priors_log = torch.log(pi).to(device)

    # Calcolo pesi di classe w_k[y] se eps > 0
    client_weights_by_class = None
    if eps is not None and eps > 0.0 and client_class_counts is not None:
        shifted_counts = client_class_counts + float(eps)
        weights = shifted_counts / shifted_counts.sum(dim=0, keepdim=True)
        client_weights_by_class = weights.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            client_logits = []
            for k, model in enumerate(client_models):
                z_k = model(images)
                if tau > 0.0 and client_priors_log is not None:
                    z_k = z_k - tau * client_priors_log[k]
                client_logits.append(z_k)

            if client_weights_by_class is not None:
                stacked = torch.stack(client_logits, dim=0)  # [K, B, C]
                w = client_weights_by_class.unsqueeze(1)     # [K, 1, C]
                teacher_logits = (stacked * w).sum(dim=0)    # [B, C]
            else:
                teacher_logits = sum(client_logits) / num_clients

            loss = criterion(teacher_logits, labels)
            total_loss += loss.item() * batch_size

            _, preds = torch.max(teacher_logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate_student_model(model, data_loader, device='cpu'):
    """
    Valuta il modello globale (studente) calcolando:
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
# 4. FUSIONE CON DISTILLAZIONE CALIBRATA (FedDF Calibrated)
# ==============================================================================
def fed_df_calibrated_fusion(global_model, client_models, unlabeled_loader, client_sizes=None,
                             tau=0.0, eps=None, client_class_counts=None,
                             epochs=3, lr=0.001, temperature=2.0, device='cpu'):
    """
    Fase di distillazione d'insieme sul server (FedDF):
    1. Warm-up con media pesata FedAvg (proporzionale a client_sizes).
    2. A ogni batch del proxy dataset non etichettato:
       - Forward di tutti i client (insegnanti congelati).
       - Correzione dei logit se tau > 0: z_k_corr = z_k - tau * log(pi_k)
       - Aggregazione per classe se eps > 0: w_k[y] = (n_k[y] + eps) / sum_j (n_j[y] + eps)
         oppure media semplice (1/K) se eps is None / <= 0.
       - Generazione target soft: p_teacher = Softmax(z_teacher / T).
       - Aggiornamento studente globale minimizzando la KL divergence scalata per T^2.
    """
    # 1. Warm-up ponderato FedAvg
    client_weights_dict = {i: model.state_dict() for i, model in enumerate(client_models)}
    avg_weights = fed_avg(client_weights_dict, client_sizes=client_sizes)
    global_model.load_state_dict(avg_weights)

    global_model.to(device)
    for model in client_models:
        model.to(device)
        model.eval()

    global_model.train()
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    num_clients = len(client_models)

    # Precalcolo prior log(pi_k)
    client_priors_log = None
    if tau > 0.0 and client_class_counts is not None:
        smooth_counts = client_class_counts + 1.0
        pi = smooth_counts / smooth_counts.sum(dim=1, keepdim=True)
        client_priors_log = torch.log(pi).to(device)

    # Precalcolo pesi per classe w_k[y]
    client_weights_by_class = None
    if eps is not None and eps > 0.0 and client_class_counts is not None:
        shifted_counts = client_class_counts + float(eps)
        weights = shifted_counts / shifted_counts.sum(dim=0, keepdim=True)
        client_weights_by_class = weights.to(device)

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0

        for images, _ in unlabeled_loader:
            images = images.to(device)

            with torch.no_grad():
                client_logits = []
                for k, model in enumerate(client_models):
                    z_k = model(images)
                    if tau > 0.0 and client_priors_log is not None:
                        z_k = z_k - tau * client_priors_log[k]
                    client_logits.append(z_k)

                if client_weights_by_class is not None:
                    stacked = torch.stack(client_logits, dim=0)  # [K, B, C]
                    w = client_weights_by_class.unsqueeze(1)     # [K, 1, C]
                    teacher_logits = (stacked * w).sum(dim=0)    # [B, C]
                else:
                    teacher_logits = sum(client_logits) / num_clients

                target_probs = F.softmax(teacher_logits / temperature, dim=1)

            optimizer.zero_grad()
            student_logits = global_model(images)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

            loss = criterion_kd(student_log_probs, target_probs) * (temperature ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        epoch_losses.append(epoch_loss / batches if batches > 0 else 0.0)

    final_kd_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    return copy.deepcopy(global_model.state_dict()), final_kd_loss


# ==============================================================================
# 5. SALVATAGGIO GRAFICI E RISULTATI
# ==============================================================================
def save_plots_and_history(history, cartella_destinazione, scenario_name):
    """
    Esporta il file CSV completo e genera la figura quadrupla ad alta risoluzione.
    """
    os.makedirs(cartella_destinazione, exist_ok=True)
    rounds = range(1, len(history['round']) + 1)

    # 1. Scrittura CSV
    csv_path = os.path.join(cartella_destinazione, 'global_history.csv')
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Round',
            'Teacher_Val_Loss', 'Teacher_Val_Acc',
            'Student_Val_Loss', 'Student_Val_Acc',
            'Distillation_Gap_Val_Acc', 'Distill_KD_Loss',
            'Student_Test_Loss', 'Student_Test_Acc'
        ] + FASHION_CLASSES
        writer.writerow(header)

        for i, r in enumerate(rounds):
            row = [
                r,
                f"{history['teacher_val_loss'][i]:.4f}",
                f"{history['teacher_val_acc'][i]:.2f}",
                f"{history['student_val_loss'][i]:.4f}",
                f"{history['student_val_acc'][i]:.2f}",
                f"{history['distill_gap_acc'][i]:.2f}",
                f"{history['distill_kd_loss'][i]:.4f}",
                f"{history['student_test_loss'][i]:.4f}",
                f"{history['student_test_acc'][i]:.2f}",
            ]
            row.extend([f"{history['test_class_acc'][i][cls]:.2f}" for cls in FASHION_CLASSES])
            writer.writerow(row)

    print(f"\n[OK] Storico numerico salvato in: {csv_path}")

    # 2. Generazione grafici ad alta risoluzione (4 pannelli)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Pannello 1: Loss curves
    ax_loss = axes[0, 0]
    ax_loss.plot(rounds, history['teacher_val_loss'], label='Teacher Val Loss', color='#e67e22', linestyle='--', linewidth=1.8)
    ax_loss.plot(rounds, history['student_val_loss'], label='Student Val Loss (ES Check)', color='#2980b9', linewidth=2.2)
    ax_loss.plot(rounds, history['student_test_loss'], label='Student Test Loss', color='#c0392b', linewidth=1.5, alpha=0.8)
    ax_loss.set_title('Loss per Round (Validation & Test)', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Round')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()

    # Pannello 2: Accuracy & Distillation Gap
    ax_acc = axes[0, 1]
    ax_acc.plot(rounds, history['teacher_val_acc'], label='Teacher Val Accuracy', color='#e67e22', linestyle='--', linewidth=2.0)
    ax_acc.plot(rounds, history['student_val_acc'], label='Student Val Accuracy', color='#27ae60', linewidth=2.2)
    ax_acc.plot(rounds, history['student_test_acc'], label='Student Test Accuracy', color='#2980b9', linewidth=2.0)
    ax_acc.set_title('Accuracy Curves & Distillation Gap', fontsize=12, fontweight='bold')
    ax_acc.set_xlabel('Round')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    ax_acc.legend()

    # Pannello 3: Distillation KL Loss
    ax_kd = axes[1, 0]
    ax_kd.plot(rounds, history['distill_kd_loss'], label='Server Distill KL Loss', color='#8e44ad', marker='^', markersize=3, linewidth=2.0)
    ax_kd.set_title('Distillation KD Loss (KL-Divergence)', fontsize=12, fontweight='bold')
    ax_kd.set_xlabel('Round')
    ax_kd.set_ylabel('Loss KD')
    ax_kd.grid(True, linestyle='--', alpha=0.6)
    ax_kd.legend()

    # Pannello 4: Per-Class Accuracy sul Test Set
    ax_cls = axes[1, 1]
    cmap = plt.get_cmap('tab10')
    for idx, cls in enumerate(FASHION_CLASSES):
        cls_vals = [history['test_class_acc'][r_idx][cls] for r_idx in range(len(rounds))]
        ax_cls.plot(rounds, cls_vals, label=cls, color=cmap(idx), linewidth=1.4, alpha=0.85)
    ax_cls.set_title('Student Test Accuracy per Classe (Fashion-MNIST)', fontsize=12, fontweight='bold')
    ax_cls.set_xlabel('Round')
    ax_cls.set_ylabel('Accuracy (%)')
    ax_cls.grid(True, linestyle='--', alpha=0.6)
    ax_cls.legend(fontsize=8, ncol=2, loc='lower right')

    plt.suptitle(f'Esperimento: {scenario_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(cartella_destinazione, 'learning_curves.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafici salvati in: {plot_path}")


# ==============================================================================
# 6. PIPELINE PRINCIPALE DI ADDESTRAMENTO
# ==============================================================================
def run_experiment(
    exp_name='01_FedDF_Classico',
    tau=0.0,
    eps=None,
    alpha=0.1,
    num_clients=5,
    local_epochs=20,
    distill_epochs=3,
    distill_lr=0.001,
    client_lr=0.01,
    temperature=2.0,
    num_rounds=100,
    patience=20,
    batch_size=64,
    seed=42,
    output_base_dir='./esperimenti_finali_FedDF'
):
    start_time = time.time()
    dest_folder = os.path.join(output_base_dir, exp_name)
    os.makedirs(dest_folder, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 75)
    print(f"=== AVVIO ESPERIMENTO: {exp_name} ===")
    print("=" * 75)
    print(f" - Cartella output:        {dest_folder}")
    print(f" - Dirichlet Alpha:        {alpha} (Eterogeneita Non-IID)")
    print(f" - Numero Client:          {num_clients}")
    print(f" - Epoche Locali Client:   {local_epochs}")
    print(f" - Parametro Correzione:   tau = {tau} {'(Disattivata)' if tau == 0.0 else '(Attiva)'}")
    print(f" - Parametro Pesatura:     eps = {eps} {'(Disattivata - Media 1/K)' if eps is None or eps <= 0 else '(Attiva)'}")
    print(f" - Epoche Distillazione:   {distill_epochs} (Proxy Set 10.000 campioni, Temp={temperature})")
    print(f" - Pazienza Early Stop:    {patience} round su Server Validation Loss")
    print(f" - Dispositivo Calcolo:    {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    # 1. Caricamento Dataset e Partizioni
    client_loaders, server_distill_loader, val_loader, test_loader = get_fashion_mnist_data(
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size,
        val_size=6000,
        num_distill_samples=10000,
        seed=seed
    )

    client_sizes = [len(client_loaders[i].dataset) for i in range(num_clients)]
    client_class_counts = extract_client_class_counts(client_loaders, num_classes=10)

    print("Distribuzione Campioni Locali per Client:")
    for k in range(num_clients):
        tot_k = int(client_class_counts[k].sum().item())
        print(f"  Client {k+1}: {tot_k} campioni | Classi a 0: {(client_class_counts[k] == 0).sum().item()}")

    # Inizializzazione Modello Globale
    torch.manual_seed(seed)
    global_model = MNISTCNN()
    global_weights = copy.deepcopy(global_model.state_dict())
    early_stopping = EarlyStopping(patience=patience)

    history = {
        'round': [],
        'teacher_val_loss': [],
        'teacher_val_acc': [],
        'student_val_loss': [],
        'student_val_acc': [],
        'distill_gap_acc': [],
        'distill_kd_loss': [],
        'student_test_loss': [],
        'student_test_acc': [],
        'test_class_acc': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_round = 0
    test_acc_at_best_val = 0.0

    # === CICLO DEI ROUND DI COMUNICAZIONE ===
    for round_idx in range(1, num_rounds + 1):
        round_t0 = time.time()
        print(f"\n{'-'*75}\n[ROUND {round_idx:02d}/{num_rounds}] In corso...\n{'-'*75}")

        # 1. Addestramento Locale Client
        client_models_list = []
        for i in range(num_clients):
            c_model = MNISTCNN()
            c_model.load_state_dict(copy.deepcopy(global_weights))

            c_model, _, c_loss, c_acc = train_client_local(
                model=c_model,
                train_loader=client_loaders[i],
                client_id=i+1,
                epochs=local_epochs,
                lr=client_lr,
                device=DEVICE
            )
            client_models_list.append(c_model)
            print(f"  -> Client {i+1} addestrato ({local_epochs} ep.) | Train Loss: {c_loss:.4f} | Train Acc: {c_acc:.2f}%")

        # 2. Valutazione Teacher Ensemble sul Server Validation Set (Pre-Distillazione)
        teacher_v_loss, teacher_v_acc = evaluate_teacher_ensemble(
            client_models=client_models_list,
            val_loader=val_loader,
            tau=tau,
            eps=eps,
            client_class_counts=client_class_counts,
            device=DEVICE
        )

        # 3. Distillazione d'Ensemble sul Server (Proxy Set)
        global_weights, distill_loss = fed_df_calibrated_fusion(
            global_model=global_model,
            client_models=client_models_list,
            unlabeled_loader=server_distill_loader,
            client_sizes=client_sizes,
            tau=tau,
            eps=eps,
            client_class_counts=client_class_counts,
            epochs=distill_epochs,
            lr=distill_lr,
            temperature=temperature,
            device=DEVICE
        )
        global_model.load_state_dict(global_weights)

        # 4. Valutazione Studente sul Server Validation Set (Post-Distillazione)
        stud_v_loss, stud_v_acc, _ = evaluate_student_model(global_model, val_loader, device=DEVICE)

        # 5. Valutazione Studente sul Test Set Globale
        stud_t_loss, stud_t_acc, stud_cls_acc = evaluate_student_model(global_model, test_loader, device=DEVICE)

        # Calcolo Distillation Gap (Teacher Acc - Student Acc)
        distill_gap = teacher_v_acc - stud_v_acc
        round_duration = time.time() - round_t0

        # Registrazione Storico
        history['round'].append(round_idx)
        history['teacher_val_loss'].append(teacher_v_loss)
        history['teacher_val_acc'].append(teacher_v_acc)
        history['student_val_loss'].append(stud_v_loss)
        history['student_val_acc'].append(stud_v_acc)
        history['distill_gap_acc'].append(distill_gap)
        history['distill_kd_loss'].append(distill_loss)
        history['student_test_loss'].append(stud_t_loss)
        history['student_test_acc'].append(stud_t_acc)
        history['test_class_acc'].append(stud_cls_acc)

        # Log su console sintetico e chiaro
        print(f"\n[SERVER ROUND {round_idx}] ({round_duration:.1f}s)")
        print(f"  Teacher Ensemble Val Acc:  {teacher_v_acc:.2f}% (Loss: {teacher_v_loss:.4f})")
        print(f"  Student Distilled Val Acc: {stud_v_acc:.2f}% (Loss: {stud_v_loss:.4f})")
        print(f"  Distillation Gap:          {distill_gap:+.2f}% | KD Loss: {distill_loss:.4f}")
        print(f"  Student Test Accuracy:     {stud_t_acc:.2f}% (Loss: {stud_t_loss:.4f})")

        # Checkpoint e Early Stopping (guidato strettamente dalla Server Validation Loss)
        is_best = early_stopping(stud_v_loss)
        torch.save(global_weights, os.path.join(dest_folder, 'globale_post_fed_ultimo_round.pt'))

        if is_best:
            best_val_loss = stud_v_loss
            best_val_acc = stud_v_acc
            best_round = round_idx
            test_acc_at_best_val = stud_t_acc
            torch.save(global_weights, os.path.join(dest_folder, 'miglior_modello_globale.pt'))
            print(f"  --> [CHECKPOINT] Nuovo Miglior Modello Salvato! (Best Val Loss: {best_val_loss:.4f})")

        if early_stopping.early_stop:
            print(f"\n[!] EARLY STOPPING attivato al Round {round_idx} (Nessun miglioramento su Val Loss per {patience} round consecutivi).")
            break

    total_training_time = time.time() - start_time
    print(f"\n=== Addestramento completato in {total_training_time/60:.1f} minuti ===")

    # Salvataggio CSV e Grafici
    save_plots_and_history(history, dest_folder, exp_name)

    # Salvataggio Configurazione e Risultati Sintetici JSON
    summary_data = {
        'experiment_name': exp_name,
        'tau': tau,
        'eps': eps,
        'alpha': alpha,
        'num_clients': num_clients,
        'local_epochs': local_epochs,
        'distill_epochs': distill_epochs,
        'temperature': temperature,
        'client_lr': client_lr,
        'distill_lr': distill_lr,
        'patience': patience,
        'seed': seed,
        'device': str(DEVICE),
        'total_rounds_executed': len(history['round']),
        'best_round': best_round,
        'best_student_val_loss': best_val_loss,
        'best_student_val_acc': best_val_acc,
        'student_test_acc_at_best_val': test_acc_at_best_val,
        'final_student_test_acc': history['student_test_acc'][-1],
        'total_duration_minutes': round(total_training_time / 60.0, 2)
    }

    summary_path = os.path.join(dest_folder, 'summary_esperimento.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"[OK] Summary JSON salvato in: {summary_path}")

    print("\n" + "=" * 75)
    print(f"=== RISULTATO FINALE [{exp_name}] ===")
    print(f" - Miglior Round (Val Loss):      {best_round}")
    print(f" - Best Student Val Accuracy:     {best_val_acc:.2f}% (Loss: {best_val_loss:.4f})")
    print(f" - Student Test Acc al Best Val:  {test_acc_at_best_val:.2f}%")
    print(f" - Student Test Acc Ultimo Round: {history['student_test_acc'][-1]:.2f}%")
    print("=" * 75 + "\n")

    return summary_data


# ==============================================================================
# 7. PARSER DA RIGA DI COMANDO
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FedDF Calibrated: Distillazione d'Ensemble con Logit Adjustment e Pesatura di Classe")

    parser.add_argument('--exp_name', type=str, default='01_FedDF_Classico',
                        help='Nome identificativo dello scenario')
    parser.add_argument('--tau', type=float, default=0.0,
                        help='Parametro tau per Logit Adjustment (0.0 = disattivato)')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='Parametro epsilon per Class Weighting (<=0.0 = media semplice 1/K)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Parametro alpha della distribuzione di Dirichlet')
    parser.add_argument('--num_clients', type=int, default=5,
                        help='Numero di client federati')
    parser.add_argument('--local_epochs', type=int, default=20,
                        help='Epoche di addestramento locale per client')
    parser.add_argument('--distill_epochs', type=int, default=3,
                        help='Epoche di distillazione sul server (proxy set)')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperatura T per la distillazione soft-label')
    parser.add_argument('--client_lr', type=float, default=0.01,
                        help='Learning rate SGD locale dei client')
    parser.add_argument('--distill_lr', type=float, default=0.001,
                        help='Learning rate Adam per la distillazione')
    parser.add_argument('--patience', type=int, default=20,
                        help='Pazienza per Early Stopping su Server Validation Loss')
    parser.add_argument('--num_rounds', type=int, default=100,
                        help='Numero massimo di round di comunicazione')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Dimensione del batch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per la riproducibilita')
    parser.add_argument('--output_dir', type=str, default='./esperimenti_finali_FedDF',
                        help='Cartella base di destinazione dei risultati')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Se eps <= 0.0, interpretiamo come None (aggregazione uniforme 1/K)
    effective_eps = args.eps if args.eps > 0.0 else None

    run_experiment(
        exp_name=args.exp_name,
        tau=args.tau,
        eps=effective_eps,
        alpha=args.alpha,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        distill_epochs=args.distill_epochs,
        distill_lr=args.distill_lr,
        client_lr=args.client_lr,
        temperature=args.temperature,
        num_rounds=args.num_rounds,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        output_base_dir=args.output_dir
    )
