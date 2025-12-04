import re
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. LOAD AND PREPARE THE DATA
# ============================================================================
# We load the IMDB movie review dataset (50,000 reviews: 25k positive, 25k negative).
# Then we take a 10,000-example subset for faster experiments.
# Labels: positive → 1, negative → 0.
# ============================================================================

df = pd.read_csv("data/IMDB Dataset.csv")

df_small = df.sample(10_000, random_state=123).reset_index(drop=True)
df_small["label"] = (df_small["sentiment"] == "positive").astype(int)

print(df_small["label"].value_counts())


# ============================================================================
# 2. TEXT CLEANING + TOKENIZATION
# ============================================================================
# The model cannot read raw English text — it needs numbers.
# Step 1: Clean the text (lowercase, remove HTML, keep only letters/spaces).
# Step 2: Tokenize the review into individual words.
# ============================================================================

def text_clean(s: str) -> str:
    """
    Clean a raw review.
    - lowercase all text
    - remove HTML tags like <br/>
    - remove punctuation
    - collapse multiple spaces into one
    """
    s = s.lower()
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    """Split a cleaned review into tokens (words)."""
    return text_clean(s).split()

# Tokenize all 10,000 reviews
tokenized = [tokenize(r) for r in df_small["review"]]


# ============================================================================
# 3. BUILD VOCABULARY + ENCODE AS NUMBERS
# ============================================================================
# We need to convert words into integers.
# - We keep only the top `max_words` most frequent words.
# - 0 is PAD (padding)
# - 1 is OOV (out-of-vocabulary)
# - 2..N represent actual words
#
# Each review is converted into a list of integers (a fixed length sequence).
# ============================================================================

from collections import Counter

max_words = 10_000  # Vocabulary limit
max_len   = 150     # Sequence length (truncate or left-pad to this)

# Count word frequency across all reviews
all_tokens = [tok for review in tokenized for tok in review]
freqs = Counter(all_tokens)

# Keep the most frequent words
most_common = freqs.most_common(max_words)

# Assign integer IDs to each word
# 0 = PAD, 1 = OOV, 2+ = actual vocab words
vocab = {w: i + 2 for i, (w, _) in enumerate(most_common)}
PAD_IDX = 0
OOV_IDX = 1
vocab_size = len(vocab) + 2  # add PAD + OOV

def encode(tokens):
    """
    Convert a list of tokens into a sequence of integers.
    - Unknown words → 1 (OOV)
    - If longer than max_len → keep last max_len tokens
    - If shorter than max_len → pad on the left with 0 (PAD)
    """
    ids = [vocab.get(t, OOV_IDX) for t in tokens]
    
    if len(ids) >= max_len:
        ids = ids[-max_len:]
    else:
        ids = [PAD_IDX] * (max_len - len(ids)) + ids
    
    return ids

# Encode all reviews
encoded = [encode(toks) for toks in tokenized]

X = torch.tensor(encoded, dtype=torch.long)
y = torch.tensor(df_small["label"].values, dtype=torch.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================================
# 4. TRAIN/TEST SPLIT
# ============================================================================
# We split the dataset into:
# - 80% training data
# - 20% test data
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=123,
    stratify=y
)


# ============================================================================
# 5. TORCH DATASET + DATALOADER
# ============================================================================
# These classes allow PyTorch to efficiently feed batches of data into the GRU.
# ============================================================================

class IMDBDataset(Dataset):
    """Simple dataset returning (input_sequence, label)."""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = IMDBDataset(X_train, y_train)
test_ds  = IMDBDataset(X_test,  y_test)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)


# ============================================================================
# 6. GRU MODEL DEFINITION
# ============================================================================
# GRU Components:
# - Embedding layer: converts word IDs → dense vectors
# - GRU: reads the sequence and learns dependencies between words
# - Linear layer: maps GRU output → prediction (positive vs negative)
#
# The final hidden state of the GRU represents the entire movie review.
# ============================================================================

class GRUSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        
        # Embedding layer turns word indices into 128-dimensional vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )
        
        # GRU reads the embedded sequence
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Fully-connected layer outputs a single probability
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        emb = self.embedding(x)      # shape: (batch, seq_len, embed_dim)
        out, h_n = self.gru(emb)     # h_n = final hidden state of GRU
        h_final = h_n[0]             # shape: (batch, hidden_dim)
        logits = self.fc(h_final)    # raw output
        return torch.sigmoid(logits) # probability in [0,1]


# Put model on CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUSentiment(vocab_size=vocab_size).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
# For each epoch:
# - Load batches of reviews
# - Run them through the GRU
# - Compute loss
# - Update weights using Adam
# ============================================================================

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total = 0
    total_loss = 0
    
    for xb, yb in train_dl:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        
        # Track accuracy during training
        predicted = (preds > 0.5).float()
        total_correct += (predicted == yb).sum().item()
        total += yb.size(0)
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | loss={total_loss/len(train_dl):.4f} | acc={total_correct/total:.4f}")


# ============================================================================
# 8. TEST ACCURACY
# ============================================================================
# After training, we evaluate on unseen test data.
# ============================================================================

model.eval()
total_correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        
        preds = model(xb)
        predicted = (preds > 0.5).float()
        
        total_correct += (predicted == yb).sum().item()
        total += yb.size(0)

print("Test accuracy:", total_correct / total)
