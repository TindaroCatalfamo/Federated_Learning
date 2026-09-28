import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Costanti di normalizzazione canoniche per FashionMNIST
FASHION_MEAN = 0.2860
FASHION_STD = 0.3530

def get_fashion_mnist_data(
    num_clients=5, 
    alpha=1.0, 
    batch_size=32, 
    val_size=6000, 
    num_distill_samples=10000, 
    seed=42
):
    """
    Partizionamento unificato e FAIR per FedAvg e FedDF su FashionMNIST:
    
    1. Dataset totale di training: 60.000 immagini.
       - Server Validation Set (val_size=6000, 10%): riservato al server per Early Stopping e Model Selection.
       - Server Distillation Set (num_distill_samples=10000): riservato al server per la distillazione FedDF (non etichettato).
       - Client Training Set (44.000 immagini): partizionato tra i 'num_clients' tramite distribuzione Dirichlet (parametro 'alpha').
       
       IMPORTANTE: Sia FedAvg che FedDF usano esattamente le STESSE partizioni di train per i client!
    
    2. Test Set Globale: 10.000 immagini per la valutazione imparziale finale.
    
    Ritorna:
        client_loaders: lista di DataLoader per ciascun client
        server_distill_loader: DataLoader per i dati di distillazione del server (FedDF)
        val_loader: DataLoader per la validazione globale del server (Early Stopping)
        test_loader: DataLoader per il test set globale
    """
    # Impostiamo il seed per la riproducibilità scientifica
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FASHION_MEAN,), (FASHION_STD,))
    ])

    # 1. Caricamento Dataset Base (FashionMNIST)
    full_train_dataset = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Divisione deterministica del Training Set (60.000 campioni)
    total_train_len = len(full_train_dataset)
    indices = np.random.permutation(total_train_len)
    
    # a) Validation set del server (10% = 6.000 campioni)
    val_indices = indices[:val_size]
    val_dataset = Subset(full_train_dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # b) Server Distillation set (FedDF, 10.000 campioni non etichettati)
    distill_end = val_size + num_distill_samples
    server_distill_indices = indices[val_size:distill_end]
    if num_distill_samples > 0:
        server_distill_dataset = Subset(full_train_dataset, server_distill_indices)
        server_distill_loader = DataLoader(server_distill_dataset, batch_size=128, shuffle=True)
    else:
        server_distill_loader = None

    # c) Client Training data (44.000 campioni rimanenti)
    client_pool_indices = indices[distill_end:]
    
    # 3. Partizionamento Dirichlet sui 44.000 campioni dei Client
    partitions_dir = './dirichlet_partitions'
    os.makedirs(partitions_dir, exist_ok=True)
    indices_file = os.path.join(
        partitions_dir, 
        f'client_idx_dirichlet_fashion_c{num_clients}_a{alpha}_s{num_distill_samples}_v{val_size}_seed{seed}.pt'
    )

    is_valid = False
    if os.path.exists(indices_file):
        try:
            client_indices_mapped = torch.load(indices_file, weights_only=False)
            if isinstance(client_indices_mapped, list) and len(client_indices_mapped) == num_clients:
                if sum(len(idx) for idx in client_indices_mapped) == len(client_pool_indices):
                    is_valid = True
        except Exception:
            is_valid = False

    if is_valid:
        print(f"-> Caricamento partizioni Dirichlet da '{indices_file}'...")
    else:
        print(f"-> Calcolo distribuzione Dirichlet per {num_clients} client (Alpha={alpha}, Client Samples={len(client_pool_indices)}, Seed={seed})...")
        client_indices_mapped = [[] for _ in range(num_clients)]
        pool_targets = np.array([full_train_dataset.targets[i] for i in client_pool_indices])
        
        for class_id in range(10):
            idx_for_class = np.where(pool_targets == class_id)[0]
            np.random.shuffle(idx_for_class)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.cumsum(proportions)
            splits = (proportions * len(idx_for_class)).astype(int)[:-1]
            
            idx_split = np.split(idx_for_class, splits)
            for i in range(num_clients):
                client_indices_mapped[i].extend(idx_split[i].tolist())

        for i in range(num_clients):
            np.random.shuffle(client_indices_mapped[i])
        torch.save(client_indices_mapped, indices_file)

    # 4. Creazione dei Dataloader per i client
    client_loaders = []
    targets_np = np.array(full_train_dataset.targets)
    print(f"\n[FASHION-MNIST FL DATASET - Data Fairness]")
    print(f"   - Validation Server (Early Stopping): {len(val_dataset)} immagini (10%)")
    if server_distill_loader is not None:
        print(f"   - Distillazione Server (FedDF): {num_distill_samples} immagini non etichettate")
    print(f"   - Dati Totali Client (condivisi tra FedAvg e FedDF): {len(client_pool_indices)} immagini")
    print(f"   - Split Dirichlet (Alpha={alpha}, Client={num_clients}):")
    
    for i, indices_mapped in enumerate(client_indices_mapped):
        final_indices = [client_pool_indices[idx] for idx in indices_mapped]
        ds = Subset(full_train_dataset, final_indices)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
        client_targets = targets_np[final_indices]
        class_counts = [int(np.sum(client_targets == c)) for c in range(10)]
        print(f"      * Client {i+1}: {len(ds)} immagini | Distribuzione Classi: {class_counts}")

    print(f"   - Test Set Globale: {len(test_dataset)} immagini.\n")

    return client_loaders, server_distill_loader, val_loader, test_loader


# Alias di compatibilità
def get_fashion_mnist_fedavg(num_clients, alpha, batch_size=32, seed=42):
    client_loaders, _, val_loader, test_loader = get_fashion_mnist_data(
        num_clients=num_clients, alpha=alpha, batch_size=batch_size, seed=seed
    )
    return client_loaders, val_loader, test_loader

def get_fashion_mnist_feddf(num_clients, alpha, batch_size=32, seed=42):
    return get_fashion_mnist_data(
        num_clients=num_clients, alpha=alpha, batch_size=batch_size, seed=seed
    )