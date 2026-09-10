"""
Extended Ordinal Regression Study: Tabular + Image Datasets
============================================================
Datasets  : TABULAR (Abalone5, Abalone10, Balance, Car, Thyroid)
           + IMAGE  (FGNET age estimation, Herlev Pap Smear)
Models    : Softmax, UnimodalPAVA, UnimodalNet, CORAL, CORN
Metrics   : Accuracy, MAE, QWK
Protocol  : 5-fold CV, pretrained CNN features for images

IMAGE DATASETS:
- FGNET: Face aging (1,002 images, ages 0-69, ~60 age classes binned to 8)
- Herlev: Cervical cell classification (917 images, 7 ordinal classes)

For images, we extract features using a pretrained ResNet-50 backbone
(frozen, trained on ImageNet), then train only the ordinal output head.
"""
import sys, warnings, copy, os
#sys.path.insert(0, '/mnt/user-data/outputs')
warnings.filterwarnings('ignore')

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score

#from callibration import k2ECE_memoryEfficient
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
def k2ECE_memoryEfficient (y,yhat,conf, bandwidth = 0.1):
    N = y.shape[0]
    lacc = np.zeros((N,1))
    for i in range(N):
        dist = pairwise_distances(conf[[i],:], conf)
        votes = norm.pdf(dist[0,:], 0, bandwidth)
        lacc_dem = np.sum(votes)
        lacc_num = np.sum(votes[y[:,0]==yhat[:,0]])
        lacc[i] = lacc_num/lacc_dem      
    kece = np.sum(np.abs(lacc-conf))/N
    return kece

# Conditional imports for vision tasks (may not be available in all envs)
try:
    import torchvision
    from torchvision import models, transforms
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("[WARNING] torchvision not available — image datasets skipped")

from unisparse import UnimodalProjectionLayer
from ord_acl_vssl import OrdACLModel, VSSLModel
from mysparsemax import sparsemax

DEVICE  = torch.device('cpu')
SEEDS   =  [42, 43, 44, 45, 46]     #[0, 1, 2, 3, 4]
EPOCHS  = 1000
LR      = 1e-4 #or 5*1e-4 for unisparse
HIDDEN  = 128 #128 192 256
N_FOLDS = 5
CLIP    = 5.0
basedir = Path("C:/Users/jaime/Desktop/unisparse")
basedir = Path(".")


print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version (torch):", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))


# ── Datasets ──────────────────────────────────────────────────────────────────
def generate_synthetic_data4D(N):
    points = np.random.uniform(0, 1, size=(N, 4))
    scores_clean = 1000 * np.prod(points - 0.5, axis=1)
    scores = scores_clean + np.random.normal(0, 0.125, np.shape(scores_clean))
    bin_edges = [-5, -2.5, -1, -0.4, 0.1, 0.5, 1.1, 3, 6]
    bin_number_clean = np.digitize(scores_clean, bin_edges)
    bin_number = np.digitize(scores, bin_edges)
    K = len(bin_edges) + 1
    return (
        points, #torch.tensor(points, dtype=torch.float32),
        bin_number, #torch.tensor(bin_number, dtype=torch.long),
        K,
        bin_number_clean, #torch.tensor(bin_number_clean, dtype=torch.long),
    )
def load_synthetic():
    X, y, K, _ = generate_synthetic_data4D(200)
    return X, y , K

def load_abalone5():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/abalone/abalone.data")
    url = (basedir /"myUCI/abalone.data")      
    cols = ['Sex','Length','Diameter','Height','WholeWeight',
            'ShuckedWeight','VisceraWeight','ShellWeight','Rings']
    df = pd.read_csv(url, names=cols)
    df = pd.get_dummies(df, columns=['Sex'], drop_first=False)
    y_raw = df.pop('Rings').values.astype(int)
    X = df.values.astype(np.float32)
    bins = [0, 6, 8, 9, 10, 100]
    y = np.clip(np.digitize(y_raw, bins) - 1, 0, 4).astype(np.int64)
    return X, y, 5

def load_abalone10():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/abalone/abalone.data")
    url = (basedir /"myUCI/abalone.data")      
    cols = ['Sex','Length','Diameter','Height','WholeWeight',
            'ShuckedWeight','VisceraWeight','ShellWeight','Rings']
    df = pd.read_csv(url, names=cols)
    df = pd.get_dummies(df, columns=['Sex'], drop_first=False)
    y_raw = df.pop('Rings').values.astype(int)
    X = df.values.astype(np.float32)
    deciles = np.unique(np.percentile(y_raw, np.linspace(0, 100, 11)))
    y = np.digitize(y_raw, deciles[1:-1]).astype(np.int64)
    return X, y, len(np.unique(y))

def load_balance_scale():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/balance-scale/balance-scale.data")
    url = (basedir /"myUCI/balance-scale.data")      
    df = pd.read_csv(url, header=None)
    y = df[0].map({'L': 0, 'B': 1, 'R': 2}).values.astype(np.int64)
    X = df.iloc[:, 1:].values.astype(np.float32)
    return X, y, 3

def load_car():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/car/car.data")
    url = (basedir /"myUCI/car.data")    
    cols = ['buying','maint','doors','persons','lug_boot','safety','class']
    df = pd.read_csv(url, names=cols)
    y = df['class'].map({'unacc':0,'acc':1,'good':2,'vgood':3}).values.astype(np.int64)
    df = pd.get_dummies(df.drop(columns=['class']), drop_first=False)
    return df.values.astype(np.float32), y, 4

def load_new_thyroid():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/thyroid-disease/new-thyroid.data")
    url = (basedir /"myUCI/new-thyroid.data")
    df = pd.read_csv(url, header=None)
    y = (df[0].values - 1).astype(np.int64)
    X = df.iloc[:, 1:].values.astype(np.float32)
    return X, y, int(y.max() + 1)




# ══════════════════════════════════════════════════════════════════════════════
# 1B. IMAGE DATASETS (FGNET, HERLEV) — LOAD AS PRETRAINED FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def load_fgnet_features():
    features_dir = Path(basedir / "fgnet_features")
    
    if not (features_dir / "fgnet_features.npy").exists():
        raise FileNotFoundError(
            "Run 'python extract_fgnet_features.py --fgnet-dir ./FG-NET' first"
        )
    
    X = np.load(features_dir / "fgnet_features.npy")
    y = np.load(features_dir / "fgnet_labels.npy")
    return X, y, 8


def load_herlev_features():
    features_dir = Path(basedir/ "herlev_features")
    
    if not (features_dir / "herlev_features.npy").exists():
        raise FileNotFoundError("See HERLEV_SETUP_GUIDE.md for setup")
    
    X = np.load(features_dir / "herlev_features.npy")
    y = np.load(features_dir / "herlev_labels.npy")
    return X, y, 7


# ── Backbone ──────────────────────────────────────────────────────────────────
class Backbone(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc   = nn.Linear(in_dim, HIDDEN)
        self.act  = nn.ReLU()
        self.drop = nn.Identity() # nn.Dropout(DROPOUT)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        return self.drop(self.act(self.fc(x)))

# ── Models ────────────────────────────────────────────────────────────────────
class SoftmaxModel(nn.Module):
    def __init__(self, in_dim, K):
        super().__init__()
        self.bb = Backbone(in_dim)
        self.h  = nn.Linear(HIDDEN, K)
        nn.init.xavier_uniform_(self.h.weight); nn.init.zeros_(self.h.bias)
    def forward(self, x): return F.softmax(self.h(self.bb(x)), dim=-1)
    def predict(self, x): 
        p =self(x) 
        return p, torch.argmax(p, dim = 1)
    def loss(self, x, y): return F.cross_entropy(self.h(self.bb(x)), y)

class sparsemaxModel(nn.Module):
    def __init__(self, in_dim, K):
        super().__init__()
        self.bb = Backbone(in_dim)
        self.h  = nn.Linear(HIDDEN, K)
        nn.init.xavier_uniform_(self.h.weight); nn.init.zeros_(self.h.bias)
    def forward(self, x): return sparsemax(self.h(self.bb(x)), dim=-1)
    def predict(self, x): 
        p =self(x) 
        return p, torch.argmax(p, dim = 1)
    def loss(self, x, y): return F.cross_entropy(self.h(self.bb(x)), y)

class StochasticUnimodalMLP(nn.Module):
    """
    MLP with stochastic normalization selection during training.
    
    Both paths use PAVA (unimodal projection), but differ in normalization:
        - With probability β:     PAVA → Sparsemax (sparse + unimodal)
        - With probability (1-β): PAVA → Softmax  (dense + unimodal)
    
    At test time:
        - Always uses PAVA → Sparsemax (deterministic)
    
    This allows studying the contribution of sparsity while maintaining
    unimodality in both training modes. Can also be used for curriculum
    learning where the model gradually transitions to sparse predictions.
    
    Args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension (default: 128)
        num_classes: number of output classes
        beta: probability of using Sparsemax during training (default: 1.0)
              beta=1.0 → always PAVA→Sparsemax (deterministic)
              beta=0.5 → 50% Sparsemax, 50% Softmax (both with PAVA)
              beta=0.0 → always PAVA→Softmax during training
        alpha: sparsemax interpolation parameter (default: 1.0)
        dropout: dropout probability (default: 0.3)
    
    Example:
        # Always sparse unimodal (standard)
        model = StochasticUnimodalMLP(input_dim=10, num_classes=5, beta=1.0)
        
        # 70% sparse, 30% dense (both unimodal)
        model = StochasticUnimodalMLP(input_dim=10, num_classes=5, beta=0.7)
        
        # Curriculum: gradually increase sparsity
        model = StochasticUnimodalMLP(input_dim=10, num_classes=5, beta=0.0)
        for epoch in range(100):
            model.set_beta(min(1.0, epoch / 50))
            train_one_epoch(...)
    
    Note:
        Both training paths apply PAVA projection for unimodality.
        The difference is only in the normalization step:
            - Sparsemax (beta path): produces sparse probabilities
            - Softmax ((1-beta) path): produces dense probabilities
        This isolates the contribution of sparsity in your ablation studies.
    """
    
#    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 10, beta: float = 1.0, alpha: float = 1.0, dropout: float = 0.3):
    def __init__(self, input_dim: int, num_classes: int = 10, beta: float = 1.0, alpha: float = 1.0, dropout: float = 0.3):        
        super().__init__()
        
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        
        self.input_dim = input_dim
        self.hidden_dim = HIDDEN #hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.alpha = alpha
        
        # Backbone MLP
        #self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU(), nn.Dropout(dropout) )
        self.backbone = Backbone(input_dim)

        # Output head (shared by both paths)
        self.head = nn.Linear(self.hidden_dim, num_classes)
        
        #
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

        # Unimodal projection layer with sparsemax
        self.unimodal_sparsemax = UnimodalProjectionLayer(
            normalization='sparsemax',
            alpha=alpha
        )
        
        # Unimodal projection layer with softmax
        self.unimodal_softmax = UnimodalProjectionLayer(
            normalization='softmax',
            alpha=alpha            
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) input features
        
        Returns:
            p: (B, num_classes) output probabilities (always unimodal)
        """
        # Shared backbone
        h = self.backbone(x)  # (B, hidden_dim)
        z = self.head(h)      # (B, num_classes) logits
        
        # Stochastic normalization selection during training
        if self.training and self.beta < 1.0:
            # Sample: should we use sparsemax or softmax?
            use_sparsemax = torch.rand(1).item() < self.beta
            
            if use_sparsemax:
                # Path 1: PAVA → Sparsemax (sparse + unimodal)
                p = self.unimodal_sparsemax(z)
            else:
                # Path 2: PAVA → Softmax (dense + unimodal)
                p = self.unimodal_softmax(z)
        else:
            # Test time or beta=1.0: always use PAVA → Sparsemax
            p = self.unimodal_sparsemax(z)
        
        return p
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: (B, input_dim) input features
        
        Returns:
            y_pred: (B,) predicted class indices
        """
        self.eval()
        with torch.no_grad():
            p = self(x)
            #return p.argmax(dim=-1)
            return p, torch.argmax(p, dim = 1)        
    
    def loss(self, x, y):
        return F.nll_loss(torch.log(self.forward(x) + 1e-12), y)
        
    def set_beta(self, new_beta: float):
        """Update beta parameter (useful for curriculum learning)."""
        if not (0.0 <= new_beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {new_beta}")
        self.beta = new_beta
    
    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'num_classes={self.num_classes}, beta={self.beta}, alpha={self.alpha}')

###############
########### UnimodalNet with sparse option
class UnimodalNetModel(nn.Module):
    def __init__(self, in_dim, K, normalization='sparsemax'):
        super().__init__()
        self.bb = Backbone(in_dim)
        self.h  = nn.Linear(HIDDEN, K)
        self.normalization = normalization        
        nn.init.xavier_uniform_(self.h.weight); nn.init.zeros_(self.h.bias)

    def _act(self, logits):
        #p  = F.relu(logits) # F.softmax(logits, dim=-1) F.softplus(logits)
        p  = F.softplus(logits) # F.softmax(logits, dim=-1) F.softplus(logits)
        b  = torch.cumsum(p, dim=-1)
        c  = torch.flip(torch.cumsum(torch.flip(p,[-1]),dim=-1),[1])
        d = torch.minimum(b, c)
        if self.normalization == 'sparsemax':        
            return sparsemax(d, dim=-1)
        else:
            return F.softmax(d, dim=-1) 
        
    def forward(self, x): return self._act(self.h(self.bb(x)))

    def predict(self, x): 
        p =self(x)
        return p, torch.argmax(p, dim = 1)      
    def loss(self, x, y):
        return F.nll_loss(torch.log(self(x) + 1e-12), y)
############### END OF
########### UnimodalNet with sparse option
#        
class CoralModel(nn.Module):
    def __init__(self, in_dim, K):
        super().__init__()
        self.K  = K
        self.bb = Backbone(in_dim)
        self.w  = nn.Linear(HIDDEN, 1, bias=False)
        self.b  = nn.Parameter(torch.zeros(K - 1))
        nn.init.xavier_uniform_(self.w.weight)
    def _logits(self, x):
        return self.w(self.bb(x)) + self.b.unsqueeze(0)  # (B, K-1)
    def forward(self, x): return self._logits(x)
    def predict(self, x):
        logits = self._logits(x)    
        pred = torch.sigmoid(logits).gt(0.5).sum(1).long()    

        lixo1 = torch.sigmoid(logits) 
        lixo2 = torch.diff(lixo1, dim = 1)
        lixo3 = 1-lixo1[:,[0]]
        lixo4 = lixo1[:,[-1]]
        prob = torch.cat((lixo3, lixo2, lixo4), dim=1) #for sparsity metric only

        return prob, pred
    def loss(self, x, y):
        logits = self._logits(x)
        lvls   = torch.stack([(y > k).float() for k in range(self.K-1)], dim=1)
        return F.binary_cross_entropy_with_logits(logits, lvls)

class CornModel(nn.Module):
    def __init__(self, in_dim, K):
        super().__init__()
        self.K  = K
        self.bb = Backbone(in_dim)
        self.h  = nn.Linear(HIDDEN, K - 1)
        nn.init.xavier_uniform_(self.h.weight); nn.init.zeros_(self.h.bias)
    def forward(self, x): return self.h(self.bb(x))
    def predict(self, x):
        logits = self(x)

        lixo1 =  torch.cumprod(torch.sigmoid(logits),dim=1)
        lixo2 = torch.diff(lixo1, dim = 1)
        lixo3 = 1-lixo1[:,[0]]
        lixo4 = lixo1[:,[-1]]
        prob = torch.cat((lixo3, lixo2, lixo4), dim=1) #for sparsity metric only

        return prob, torch.cumprod(torch.sigmoid(logits),dim=1).gt(0.5).sum(1).long()
    
    def loss(self, x, y):
        logits = self(x)
        total, n = torch.zeros(1, device=logits.device), 0
        for k in range(self.K - 1):
            mask = y >= k
            if mask.sum() == 0: continue
            tgt  = (y[mask] >= k + 1).float()
            total = total + F.binary_cross_entropy_with_logits(logits[mask, k], tgt)
            n += 1
        return total / max(n, 1)



MODELS = {
##    'Softmax':      SoftmaxModel,
##    'Sparsemax':    sparsemaxModel,

##    'CORAL':        CoralModel,
##    'CORN':         CornModel,

##    'ORD-ACL':      lambda in_dim, K: OrdACLModel(Backbone(in_dim), K, HIDDEN, rho_fn='exp'),
##    'VS-SL':        lambda in_dim, K: VSSLModel(Backbone(in_dim), K, HIDDEN, rho_fn='exp', tau_fn='abs'),

##    'UnimodalNet':   lambda in_dim, K: UnimodalNetModel(in_dim, K, "softmax"),    #Pure unimodalnet
##    'UnimodalNetsparse':   lambda in_dim, K: UnimodalNetModel(in_dim, K, "sparsemax"),    #Pure unimodalnet    

##    'UnimodalPAVA': UnimodalPAVAModel, #softmax during training without PAVA
##    'StochasticUnimodalMLPa1b1': lambda in_dim, K: StochasticUnimodalMLP(in_dim, K, 1.0, 1.0),    #Pure unisparse
    'StochasticUnimodalMLPb0a1': lambda in_dim, K: StochasticUnimodalMLP(in_dim, K, 0.0, 1.0),    # always PAVA→Softmax during training; pure unisparse during testing 
##    'StochasticUnimodalMLPb0a0': lambda in_dim, K: StochasticUnimodalMLP(in_dim, K, 0.0, 0.0),    # logits->softmax during training ; pure unisparse during testing    
##    'StochasticUnimodalMLPa099b1': lambda in_dim, K: StochasticUnimodalMLP(in_dim, K, 1, 0.99),    #interpolation with 1% of the original logit during training  
}

DATASETS = {}
DATASETS['NewThyroid']=   load_new_thyroid  
#DATASETS['BalanceScale']= load_balance_scale
#DATASETS['Car'] = load_car    
##DATASETS['Synthetic']= load_synthetic
#DATASETS['Abalone5']= load_abalone5
#DATASETS['Abalone10']= load_abalone10
# Add image datasets if available
if VISION_AVAILABLE:
    DATASETS['FGNET'] = load_fgnet_features
    DATASETS['Herlev'] = load_herlev_features


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(yt, yp, K, probs):
    acc = float((yt == yp).mean())
    mae = float(np.abs(yt.astype(float) - yp.astype(float)).mean())
    sparsity = (probs < 1e-5).astype(float).mean()*K/(K-1) #from 0 to 100

    ece = k2ECE_memoryEfficient (np.reshape(yt, (-1,1)), np.reshape(yp, (-1,1)), np.max(probs, axis = 1, keepdims=True))

    try:    qwk = float(cohen_kappa_score(yt, yp, labels=list(range(K)), weights='quadratic'))
    except: qwk = float('nan')
    return acc, mae, qwk, sparsity, ece

# ── Training ──────────────────────────────────────────────────────────────────
def _bs(n): return 32 if n<200 else (64 if n<1000 else 128)

def train_fold(model_cls, in_dim, K, X_tr, y_tr, X_va, y_va, seed):
    torch.manual_seed(seed)
    m    = model_cls(in_dim, K).to(DEVICE)

    # Move full validation tensors once (cheaper than doing it every epoch)
    X_va = X_va.to(DEVICE)
    y_va = y_va.to(DEVICE)


    opt  = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=30, min_lr=1e-6)
    ldr  = DataLoader(TensorDataset(X_tr, y_tr),
                      batch_size=_bs(X_tr.shape[0]), shuffle=True)
    best_val, best_state = float('inf'), None
    for epoch in range(EPOCHS):
        m.train()
        for xb, yb in ldr:

            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad()
            loss = m.loss(xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step()
            m.eval()
        with torch.no_grad():
            vl = m.loss(X_va, y_va).item()
            probs, yp = m.predict(X_va)
            _, b, _, _, _ =metrics(yp.detach().cpu().numpy(), y_va.detach().cpu().numpy(), K, probs.detach().cpu().numpy())
            vl = b #MAE

        if epoch % 350 == 0:
            print(f"  epoch {epoch:3d}/{EPOCHS} validation loss={vl:.4f}")            
        sch.step(vl)
        if vl < best_val:
            #print (f"best epoch {epoch:3d} with loss ", vl)
            best_val   = vl
            best_state = copy.deepcopy(m.state_dict()) 
    m.load_state_dict(best_state)
    m.eval()
    return m

# ── CV ────────────────────────────────────────────────────────────────────────
def run_cv(X, y, K, model_cls, seed):
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_m = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        Xtv, Xte = X[tr_idx], X[te_idx]
        ytv, yte = y[tr_idx], y[te_idx]
        vs = max(1, len(tr_idx) // N_FOLDS)
        Xtr_np, Xva_np = Xtv[:-vs], Xtv[-vs:]
        ytr_np, yva_np = ytv[:-vs], ytv[-vs:]
        print ('fold', fold, " training set size ", np.size(ytr_np))
        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_np).astype(np.float32)
        Xva_s = sc.transform(Xva_np).astype(np.float32)
        Xte_s = sc.transform(Xte.astype(np.float32))
        Xtr_t = torch.tensor(Xtr_s,  dtype=torch.float32)
        ytr_t = torch.tensor(ytr_np,  dtype=torch.long)
        Xva_t = torch.tensor(Xva_s,  dtype=torch.float32)
        yva_t = torch.tensor(yva_np,  dtype=torch.long)
        Xte_t = torch.tensor(Xte_s,  dtype=torch.float32)
        m     = train_fold(model_cls, Xtr_t.shape[1], K,
                           Xtr_t, ytr_t, Xva_t, yva_t, seed + fold*100)
        with torch.no_grad():
            Xte_t = Xte_t.to(DEVICE)
            probs, yp = m.predict(Xte_t)
        fold_m.append(metrics(yte, yp.detach().cpu().numpy(), K, probs.detach().cpu().numpy()))
    a, e, q, s, ece = zip(*fold_m)
    return float(np.mean(a)), float(np.mean(e)), float(np.mean(q)),  float(np.mean(s)), float(np.mean(ece)),\
           float(np.std(a)),  float(np.std(e)),  float(np.std(q)),  float(np.std(s)),  float(np.std(ece)),

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    rows = []
    for ds_name, loader in DATASETS.items():
        print(f"\n{'='*65}\nDataset: {ds_name}")
        try:   X, y, K = loader()
        except Exception as e: print(f"  [SKIP] {e}"); continue
        uniq = np.unique(y)
        remap = {v:i for i,v in enumerate(uniq)}
        y = np.array([remap[v] for v in y], dtype=np.int64)
        K = len(uniq)
        print(f"  n={len(y)}, K={K}, dist={np.bincount(y).tolist()}")
        for mname, mcls in MODELS.items():
            print (mname)
            start_time = time.time()
            sa, se, sq, ss, sece = [], [], [], [], []
            for seed in SEEDS:
                try:
                    a, e, q, s, ece, *_ = run_cv(X, y, K, mcls, seed)
                    sa.append(a); se.append(e); sq.append(q); ss.append(s), sece.append(ece)
                except Exception as ex:
                    import traceback; traceback.print_exc()
            elapsed_time = time.time() - start_time
            print ("ELAPSED TIME ", elapsed_time)
            if sa:
                ma,me,mq, ms, mece = np.mean(sa),np.mean(se),np.mean(sq), np.mean(ss), np.mean(sece)
                da,de,dq, ds, dece = np.std(sa), np.std(se), np.std(sq), np.std (ss), np.std (sece)
                print(f"  {mname:15s}  Acc={ma:.3f}±{da:.3f}  "
                      f"MAE={me:.3f}±{de:.3f}  QWK={mq:.3f}±{dq:.3f} SPARSITY={ms:.3f}±{ds:.3f} ECE={mece:.3f}±{dece:.3f}")
                rows.append({'Dataset':ds_name,'Model':mname,
                    'Acc':f"{ma:.3f} ± {da:.3f}",
                    'MAE':f"{me:.3f} ± {de:.3f}",
                    'QWK':f"{mq:.3f} ± {dq:.3f}",
                    'SPT':f"{ms:.3f} ± {ds:.3f}",
                    'ECE':f"{mece:.3f} ± {dece:.3f}",                    
                    'TIME':f"{elapsed_time:.3f}", 
                    '_acc':ma,'_mae':me,'_qwk':mq, '_spt':ms, '_ece':mece, '_time':elapsed_time})
    return pd.DataFrame(rows)

def mark_best(df):
    for ds in df['Dataset'].unique():
        m = df['Dataset']==ds
        for raw, col, hi in [('_acc','Acc',True),('_mae','MAE',False),('_qwk','QWK',True),('_spt','SPT',True),('_ece','ECE',False)]:
            v = df.loc[m, raw].values.astype(float)
            b = v.max() if hi else v.min()
            for idx in df.loc[m].index[np.isclose(v, b)]:
                df.loc[idx, col] += '' # ' *'
    return df

if __name__ == '__main__':
    df = run()
    display = mark_best(df).drop(columns=['_acc','_mae','_qwk', '_spt', '_ece'])
    print("\n\n" + "="*75)
    print("RESULTS  (mean ± std over 5 seeds × 5-fold CV,  * = best per metric)")
    print("="*75)
    print(display.to_string(index=False))
    path = Path (basedir / 'ordinal_results-256.csv')
    display.to_csv(path, index=False)
    print(f"\nSaved → {path}")
