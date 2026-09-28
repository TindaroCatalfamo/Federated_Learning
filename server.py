import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

def fed_avg(client_weights_dict, client_sizes=None):
    """
    Aggregazione pesi FedAvg (McMahan et al.):
    - Se 'client_sizes' è specificato (dict o lista del numero di campioni per client),
      esegue la media pesata proporzionale alla dimensione del dataset locale:
      w_global = sum( (n_k / n_totale) * w_k )
    - Se 'client_sizes' è None, esegue la media aritmetica semplice (1/K per ciascun client).
    """
    client_names = list(client_weights_dict.keys())
    first_client = client_names[0]
    layer_names = client_weights_dict[first_client].keys()
    
    global_weights = copy.deepcopy(client_weights_dict[first_client])
    
    # Calcolo dei pesi di aggregazione (ponderati o uniformi)
    if client_sizes is not None:
        if isinstance(client_sizes, dict):
            total_samples = sum(client_sizes[k] for k in client_names)
            weights = {k: client_sizes[k] / total_samples for k in client_names}
        elif isinstance(client_sizes, list):
            total_samples = sum(client_sizes)
            weights = {client_names[i]: client_sizes[i] / total_samples for i in range(len(client_names))}
        else:
            raise ValueError("client_sizes deve essere un dict o una lista.")
    else:
        num_clients = len(client_names)
        weights = {k: 1.0 / num_clients for k in client_names}
        
    for layer in layer_names:
        global_weights[layer] = torch.zeros_like(global_weights[layer])
        for client in client_names:
            global_weights[layer] += client_weights_dict[client][layer] * weights[client]
            
    return global_weights


def fed_df_fusion(global_model, client_models, unlabeled_loader, client_sizes=None, epochs=1, lr=0.001, temperature=2.0, device='cpu', return_loss=False):
    """
    Ensemble Distillation for Robust Model Fusion (FedDF - Lin et al., NeurIPS 2020):
    1. Inizializzazione (Warm-up): FedAvg ponderato dei modelli ricevuti dai client.
    2. Distillazione d'Ensemble: addestramento del modello studente globale sui logit mediati (AVGLOGITS)
       dei modelli insegnanti dei client su dati non etichettati del server.
    Se return_loss=True, restituisce (weights, avg_kd_loss).
    """
    # 1. Warm-up con FedAvg ponderato dei pesi dei client sul modello globale
    client_weights_dict = {i: model.state_dict() for i, model in enumerate(client_models)}
    avg_weights = fed_avg(client_weights_dict, client_sizes=client_sizes)
    global_model.load_state_dict(avg_weights)

    global_model.to(device)
    
    for model in client_models:
        model.to(device)
        model.eval()
        
    global_model.train()
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    final_kd_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for images, _ in unlabeled_loader:
            images = images.to(device)
            
            with torch.no_grad():
                # AVGLOGITS: media aritmetica dei logit predetti da ciascun modello insegnante
                logits_list = [model(images) for model in client_models]
                avg_logits = sum(logits_list) / len(client_models)
                target_probs = F.softmax(avg_logits / temperature, dim=1)
                
            optimizer.zero_grad()
            student_logits = global_model(images)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        
            # KL divergence loss scalata per tau^2
            loss = criterion(student_log_probs, target_probs) * (temperature ** 2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        final_kd_loss = epoch_loss / len(unlabeled_loader)
        print(f"[Distillation] Epoca {epoch+1}/{epochs} | KL Loss: {final_kd_loss:.4f}")
        
    weights = copy.deepcopy(global_model.state_dict())
    if return_loss:
        return weights, final_kd_loss
    return weights


def attention_transfer_loss(student_att_norm, teacher_att_norm):
    """
    Attention Transfer Loss ufficiale (Zagoruyko & Komodakis, ICLR 2017 - Eq. 2):
    L_AT = (1 / 2) * || Q_S / ||Q_S||_2 - Q_T / ||Q_T||_2 ||_2^2
    Calcolata come somma degli scarti quadratici sui pixel per sample, mediata sul batch.
    Evita la divisione per H*W (196) introdotta dal default reduction='mean' di nn.MSELoss().
    """
    return 0.5 * (student_att_norm - teacher_att_norm).pow(2).sum(dim=-1).mean()


def fed_df_fusion_attention(global_model, client_models, unlabeled_loader, client_sizes=None, 
                            epochs=1, lr=0.001, temperature=2.0, beta=10.0, device='cpu'):
    """
    Ensemble Distillation con Activation-Based Attention Transfer (FedDF + Attention Transfer):
    (Rif. Lin et al. NeurIPS 2020 + Zagoruyko & Komodakis, ICLR 2017)
    
    1. Warm-up: FedAvg ponderato dei pesi dei modelli client sul modello globale.
    2. Distillazione d'Ensemble:
       - Teachers (Client): estraggono sia logits che mappe di attenzione spaziali (return_attention=True).
       - Media d'ensemble: calcolo della distribuzione soft media (KL target) e della mappa di attenzione media target.
       - Normalizzazione L2: normalizzazione vettoriale delle mappe di attenzione per ciascuna immagine.
       - Studente (Global): forward con return_attention=True.
       - Loss Totale: Loss_KD + beta * Loss_Attention (MSE tra mappe normalizzate).
       
    Restituisce:
       global_weights, avg_total_loss, avg_kd_loss, avg_at_loss
    """
    # 1. Warm-up con FedAvg ponderato dei modelli client
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
    
    avg_total_loss, avg_kd_loss, avg_at_loss = 0.0, 0.0, 0.0
    
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_at_loss = 0.0
        
        for images, _ in unlabeled_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # --- 1. FORWARD TEACHERS (CLIENT MODELS) ---
            with torch.no_grad():
                client_logits_list = []
                client_att_list = []
                for model in client_models:
                    c_logits, c_att = model(images, return_attention=True)
                    client_logits_list.append(c_logits)
                    client_att_list.append(c_att)
                
                # Media aritmetica semplice dei logit (identica a FedDF standard)
                avg_logits = sum(client_logits_list) / len(client_models)
                target_probs = F.softmax(avg_logits / temperature, dim=1)
                
                # Media aritmetica semplice della mappa di attenzione dell'ensemble: (B, 14, 14) pre-pooling
                avg_att = sum(client_att_list) / len(client_models)
                    
                # Normalizzazione L2 vettoriale della mappa target dell'ensemble (B, 196)
                target_att_flat = avg_att.view(batch_size, -1)
                target_att_norm = F.normalize(target_att_flat, p=2, dim=1)
                
            # --- 2. FORWARD STUDENTE (MODELLO GLOBALE) ---
            optimizer.zero_grad()
            student_logits, student_att = global_model(images, return_attention=True)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            # Normalizzazione L2 vettoriale della mappa dello studente
            student_att_flat = student_att.view(batch_size, -1)
            student_att_norm = F.normalize(student_att_flat, p=2, dim=1)
            
            # --- 3. CALCOLO LOSSES ---
            # KD Loss (Hinton: KL scalata per tau^2)
            loss_kd = criterion_kd(student_log_probs, target_probs) * (temperature ** 2)
            
            # Attention Transfer Loss ufficiale (Zagoruyko & Komodakis ICLR 2017: norma L2 al quadrato / 2 per sample)
            loss_at = attention_transfer_loss(student_att_norm, target_att_norm)
            
            # Loss Totale combinata
            loss_total = loss_kd + (beta * loss_at)
            
            loss_total.backward()
            optimizer.step()
            
            epoch_total_loss += loss_total.item()
            epoch_kd_loss += loss_kd.item()
            epoch_at_loss += loss_at.item()
            
        num_batches = len(unlabeled_loader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_kd_loss = epoch_kd_loss / num_batches
        avg_at_loss = epoch_at_loss / num_batches
        
        print(f"[Distillation AT] Epoca {epoch+1}/{epochs} | Total Loss: {avg_total_loss:.4f} (KD: {avg_kd_loss:.4f}, AT (x{beta}): {beta*avg_at_loss:.4f}, Raw AT: {avg_at_loss:.6f})")
        
    return copy.deepcopy(global_model.state_dict()), avg_total_loss, avg_kd_loss, avg_at_loss


def fed_df_fusion_multi_teacher_attention(global_model, client_models, unlabeled_loader, client_sizes=None, 
                                           epochs=1, lr=0.001, temperature=2.0, beta=1000.0, device='cpu'):
    """
    Multi-Teacher Attention Transfer per FedDF:
    Invece di fondere le mappe dei client in un'unica media a monte (che causa l'effetto sfocatura/blurring),
    ciascun modello client (Teacher) calcola la propria mappa di attenzione, viene normalizzato a norma L2
    individualmente, e lo Studente calcola la distanza MSE verso ciascun Teacher separatamente:
       L_AT = (1 / K) * sum_{k=1}^K MSE(Q_S, Q_k)
       
    Contemporaneamente esegue la distillazione FedDF sui logit mediati (AVGLOGITS) dell'ensemble.
    
    Restituisce:
       global_weights, avg_total_loss, avg_kd_loss, avg_at_loss
    """
    # 1. Warm-up con FedAvg ponderato dei modelli client
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
    
    num_teachers = len(client_models)
    avg_total_loss, avg_kd_loss, avg_at_loss = 0.0, 0.0, 0.0
    
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_at_loss = 0.0
        
        for images, _ in unlabeled_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # --- 1. FORWARD TEACHERS (CLIENT MODELS) ---
            with torch.no_grad():
                client_logits_list = []
                client_att_norms = []
                for model in client_models:
                    c_logits, c_att = model(images, return_attention=True)
                    client_logits_list.append(c_logits)
                    
                    # Normalizzazione L2 individuale per ciascun Teacher (B, 49)
                    c_att_flat = c_att.view(batch_size, -1)
                    client_att_norms.append(F.normalize(c_att_flat, p=2, dim=1))
                
                # Media aritmetica semplice dei logit (soft targets per FedDF)
                avg_logits = sum(client_logits_list) / num_teachers
                target_probs = F.softmax(avg_logits / temperature, dim=1)
                
            # --- 2. FORWARD STUDENTE (MODELLO GLOBALE) ---
            optimizer.zero_grad()
            student_logits, student_att = global_model(images, return_attention=True)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            # Normalizzazione L2 della mappa dello studente (B, 49)
            student_att_flat = student_att.view(batch_size, -1)
            student_att_norm = F.normalize(student_att_flat, p=2, dim=1)
            
            # --- 3. CALCOLO DELLE LOSSES ---
            # a) KD Loss classica di FedDF
            loss_kd = criterion_kd(student_log_probs, target_probs) * (temperature ** 2)
            
            # b) Multi-Teacher Attention Transfer Loss ufficiale (Zagoruyko & Komodakis ICLR 2017):
            loss_at = sum(attention_transfer_loss(student_att_norm, c_norm) for c_norm in client_att_norms) / num_teachers
            
            # c) Loss Totale Combinata
            loss_total = loss_kd + (beta * loss_at)
            
            loss_total.backward()
            optimizer.step()
            
            epoch_total_loss += loss_total.item()
            epoch_kd_loss += loss_kd.item()
            epoch_at_loss += loss_at.item()
            
        num_batches = len(unlabeled_loader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_kd_loss = epoch_kd_loss / num_batches
        avg_at_loss = epoch_at_loss / num_batches
        
        print(f"[Multi-Teacher AT] Epoca {epoch+1}/{epochs} | Total Loss: {avg_total_loss:.4f} (KD: {avg_kd_loss:.4f}, AT (x{beta}): {beta*avg_at_loss:.4f}, Raw AT: {avg_at_loss:.6f})")
        
    return copy.deepcopy(global_model.state_dict()), avg_total_loss, avg_kd_loss, avg_at_loss


def evaluate_model(model, data_loader, device='cpu'):
    """
    Valutazione standard di loss CrossEntropy e Accuratezza globale su un DataLoader.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = test_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
