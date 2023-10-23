#%%
import torch
import torch.backends.cudnn as cudnn
import dataset
from model import MatTransformer
from torch.nn.utils.rnn import pad_sequence
import wandb
import random
import tokenizer
from torch.cuda.amp import autocast, GradScaler

#%%
torch.manual_seed(42)
random.seed(42)
#%%

#%%
max_len = 500
vocab_size = 20000
version = 4

# hyperparameters 
batch_size = 8
n_embd = 512
epochs = 5
ff_size = 1024
learning_rate = 0.001
dropout_rate = 0.3
n_layer = 1
n_heads = 4

tk = (tokenizer.LangTokenizer()).load()
ds = dataset.LangDataset()
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
m = MatTransformer(max_len, n_embd, vocab_size, ff_size, dropout_rate, n_heads, n_layer, device=device)
m = m.to(device)
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
scaler = GradScaler()

total_params = sum(p.numel() for p in m.parameters())

# wandb tracking

""" wandb.init(
    # set the wandb project where this run will be logged
    project="tiny-story-v2",
    
    # track hyperparameters and run metadata
    config= {
    "learning_rate": learning_rate,
    "architecture": "MatTransformer - 4 masked blocks & Adam",
    "parameters": total_params,
    "vocab_size": vocab_size,
    "ff_size": ff_size,
    "batch_size": batch_size,
    "dropout_rate": dropout_rate,
    "dataset": "Tiny Story - 888",
    "epochs": epochs
    }
)
 """
for epoch in range(epochs):
    for idx, batch in enumerate(dl):
        c = batch['contx'].to(device)
        x = batch['input'].to(device)
        y = batch['label'].to(device)
        with autocast():
            p = m(c,x)
            p_class = p.permute(0, 2, 1)
            l = torch.nn.functional.cross_entropy(p_class, y)
        if idx % 1000 == 0: print(f"Loss: {l.item():.4f}")
        if idx % 5000 == 0: torch.save(m.state_dict(), f"weights_{epoch}_{idx}.pt")
        scaler.scale(l).backward() 
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
            
#    p = p.view(-1, p.size(-1))
#    y = y.view(-1)
        

# wandb.finish()
torch.save(m, f'models/transformer_00{version}_final.pth')
# %%