import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
from collections import Counter 
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader = None, epochs = 5, lr = 1e-3, 
                device = 'cuda', printout = True):
    """
    Run the training loop for a given model.

    Parameters:
    model (IMDbSentimentLSTM): model instance for training.
    train_loader (DataLoader): training data loader.
    val_loader (DataLoader): validation data loader.
    epochs (int): number of training epochs.
    lr (float): learning rate for optimizer.
    device (str): device on which training is run. 'cpu' or 'cuda'.
    printout (bool): If True, prints the results after each epoch.

    Returns:
    history (dict): training and validation loss and accuracy. keys (str) are 
        ['train_loss', 'train_acc', 'val_loss', 'val_acc'] and values (float) 
        are corresponding values.
    """
    model = model.to(device) 
    criterion = torch.nn.BCEWithLogitsLoss() # binary criterion
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    history = [] 

    for epoch in range(epochs):
        model.train()
        total_loss = 0 
        correct, total = 0, 0 

        pbar_train = tqdm(train_loader, leave = False, 
                          desc = f"Epoch {epoch + 1} / {epochs}")
        for xb, yb in pbar_train:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            # compute loss
            loss = criterion(outputs, yb)
            loss.backward() 
            optimizer.step() 
            total_loss += loss.item()
            # compute accuracy 
            preds = (torch.sigmoid(outputs) >= 0.5 ).float()
            correct += (preds.float() == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total 
        avg_loss = total_loss / len(train_loader) 
        if printout:
            print(f"Epoch {epoch + 1}: loss = {total_loss:.4f}, acc = {train_acc:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    val_correct += (preds == yb).sum().item()
                    val_total += yb.size(0)

            val_acc = val_correct / val_total
            val_avg_loss = val_loss / len(val_loader)
            if printout:
                print(f"\tVal loss={val_avg_loss:.4f}, Val acc={val_acc:.4f}")
        else:
            val_acc = None
            val_avg_loss = None

        history.append({'train_loss': avg_loss,
                        'train_acc': train_acc,
                        'val_loss': val_avg_loss,
                        'val_acc': val_acc
                       })
    return history

class IMDbDataset(Dataset):
    def __init__(self, data, max_len, word2idx = None):
        """ 
        Initializes the dataset from a Pandas DataFrame.

        Parameters:
        data (pd.DataFrame): Pandas DataFrame with 'review' and 'label' columns.
        max_len (int): Maximum length of the tensors for each review.
        word2idx (dict): word to index dataset, or None to build the dataset 
            from scratch
        """
        self.texts = data['review'].to_list() 
        self.labels = data['label'].tolist() 
        vocab_counter = build_vocab(self.texts)
        if word2idx is None:
            word2idx = build_word2idx(vocab_counter)
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts) 

    def __getitem__(self, idx):
        """
        Gets the encoded + padded data and label for index idx. 

        Parameters:
        idx (int): index into self.texts and self.labels.

        Returns:
        padded (torch.tensor, torch.long): padded and encoded review data.
        label (torch.tensor, torch.float): label.
        """
        text = self.texts[idx] 
        label = self.labels[idx] 

        # Encode and pad 
        encoded = encode_text(text, self.word2idx) 
        padded = pad_sequence(encoded, self.max_len) 
        padded = torch.tensor(padded, dtype = torch.long)
        label = torch.tensor(label, dtype = torch.float)
        return padded, label

class IMDbSentimentLSTM(nn.Module):
    def __init__(self, word2idx, embed_dim = 100, hidden_dim = 128,
                 num_layers = 1, dropout = 0.0, glove_path = None, 
                 glove = False, bidirectional = False, pool = False):
        """
        IMDb review sentiment LSTM.

        Parameters:
        word2idx (dict): keys (str) are words and values (int) are indices. 
        embed_dim (int): dimension of each embedding vector.
        hidden_dim (int): number of hidden units in the LSTM.
        num_layers (int): number of stackedLSTM layers.
        dropout (float): Dropout probability after LTSM.
        glove_path (str): path to the glove file.
        glove (bool): If True, uses GloVe weights. Else does not use weights.
        bidirectional (bool): If True, uses a bidirectional LSTM.
        pool (bool): If True, uses mean pooling
        """
        super().__init__() 
        self.embedding = build_embedding_matrix(word2idx, 
                                                glove_path = glove_path, 
                                                glove = glove, 
                                                embed_dim = embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first = True, bidirectional = bidirectional)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        self.pool = pool

    def forward(self, x):
        """
        LST sentiment classifier forward pass.

        Parameters:
        x (Tensor): input tensor of shape (batch_size, sequence_len) containing
            word indices.

        Returns:
        (Tensor): output tensor of shape (batch_size,) with sentiment logits.
        """
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        if self.pool:
            pooled = output.mean(dim = 1)
            dropped = self.dropout(pooled)
        else:
            if self.bidirectional:
                idx = (self.num_layers - 1) * self.num_directions
                forward_hidden = hidden[idx]
                backward_hidden = hidden[idx + 1]
                last_hidden = torch.cat((forward_hidden, backward_hidden), 
                                        dim = 1)
            else:
                last_hidden = hidden[self.num_layers - 1] 
            dropped = self.dropout(last_hidden)
        out = self.fc(dropped)
        return out.squeeze(1)

################################################################################
############################## Utility Functions ###############################
################################################################################
def create_loaders(data_train, max_len = 200, batch_size = 32, 
                   train_fraction = 0.8):
    """
    Creates training and validation loaders.

    Parameters:
    data (pd.DataFrame): Pandas DataFrame with 'review' and 'label' columns.
    max_len (int): Maximum length of the tensors for each review.
    batch_size (int): DataLoader batch size.
    train_fraction (float): fraction of data to include in the training dataset,
        with the remainder included in the validation dataset.

    Returns:
    train_loader (DataLoader): training data loader 
    val_loader (DataLoader): validation data loader, or None if 
        train_fraction = 1
    word2idx (dict): keys (str) are words and values (int) are indices. 
    """
    if (train_fraction <= 0) or (train_fraction > 1):
        raise ValueError('train_fraction must be in (0, 1]')

    train_dataset = IMDbDataset(data_train, max_len = max_len)
    
    if train_fraction == 1:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, 
                                  shuffle = True)
        return train_loader, None, train_dataset.word2idx
    
    train_size = int(train_fraction * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, 
                                                            val_size]) 
    train_loader = DataLoader(train_subset, batch_size = batch_size,
                              shuffle = True)
    val_loader   = DataLoader(val_subset  , batch_size = batch_size,
                              shuffle = False) 
    return train_loader, val_loader, train_dataset.word2idx

def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings from file into a dict.

    Parameters:
    glove_path (str): path to the glove file
    
    Returns:
    glove_dict (dict): keys (str) are words and values (??) are glove vectors 
    embed_dim (int): size of each vector
    """
    glove_dict = {} 
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(v) for v in values[1:]], 
                                  dtype = torch.float)
            glove_dict[word] = vector
    embed_dim = len(next(iter(glove_dict.values())))
    return glove_dict, embed_dim 

def build_embedding_matrix(word2idx, embed_dim = 100, glove_path = None, 
                           glove = True):
    """
    Build the PyTorch embedding layer with optional weights initialized from
        GLoVe data

    Parameters:
    word2idx (dict): keys (str) are words and values (int) are indices. 
    embed_dim (int): embedding dimension, if not glove
    glove_path (str): path to the glove file
    glove (bool): If True, uses GloVe weights. Else does not use weights

    Returns:
    (nn.Embedding): PyTorch embedding layer
    """
    vocab_size = len(word2idx)
    if not glove:
        return nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
    glove_dict, embed_dim = load_glove_embeddings(glove_path)
    matrix = torch.zeros((vocab_size, embed_dim)) 

    for word, idx in word2idx.items():
        if word in glove_dict:
            matrix[idx] = glove_dict[word]
        else:
            matrix[idx] = torch.randn(embed_dim) * 0.1 
    return nn.Embedding.from_pretrained(matrix, freeze = False, padding_idx = 0)
    
def build_vocab(text_list):
    """
    Builds a counter of vocab from a list of reviews. 

    Parameters:
    text_list (list of str): each item is a IMDb review. 

    Returns:
    counter (collections.Counter): word frequency counter. 
    """ 
    counter = Counter() 
    for text in text_list:
        tokens = text.lower().split() 
        counter.update(tokens)
    return counter 

def build_word2idx(vocab_counter, max_vocab_size = 10000):
    """
    Builds a dictionary that takes vocab words as keys and returns 
    their integer keys. 

    Parameters:
    vocab_counter (Counter): word frequency counter. 
    max_vocab_size (int): maximum number of words to include in word2idx, sorted
        by most frequent to least frequent. 

    Returns:
    word2idx (dict): keys (str) are words and values (int) are indices. 
        key "<PAD>" is used to pad sequences to the same length and key "<UNK>" 
        sets the index for words that are not in the vocab. 
    """ 
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    most_common = vocab_counter.most_common(max_vocab_size)
    for idx, (word, count) in enumerate(most_common):
        word2idx[word] = idx + 2
    return word2idx

def encode_text(text, word2idx):
    """
    Encodes text into word indices. 

    Parameters:
    text <str>: review string
    word2idx <dict>: keys (str) are words and values (int) are word indices. 
        Must include "<UNK>" key for words that do not exist. 

    Returns:
    encoded (list of int): list of word indices in text.    
    """ 
    tokens = text.lower().split()
    encoded = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens] 
    return encoded

def pad_sequence(sequence, max_len, pad_value = 0):
    """
    Ensures that the sequence has length max_len by either padding or cutting. 

    Parameters:
    sequence (list of int): sequence to cut or pad.
    max_len (int): length to which the length of sequence is padded or cut.
    pad_value (int): value to append to sequence for padding. 

    Returns:
    (list of int): sequence padded or cut to max_len. 
    """
    if len(sequence) < max_len:
        return sequence + [pad_value] * (max_len - len(sequence)) 
    return sequence[:max_len]
    
def seed_everything(seed = 4):
    """
    Sets random seeds for reproducibility.

    Parameters:
    seed (int): seed to use for random, numpy and torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
################################### Plotting ###################################
################################################################################

def plot_training(history):
    """
    Plots the training loss and accuracy history 

    Parameters:
    history (list): values (dict) are dictionaries with keys (str) in 
        ['train_loss', 'val_loss', 'train_acc', 'val_acc'] and values (float) 
        corresponding to the respective loss or accuracy

    Returns:
    fig, axs: pyplot figure and list of axes
    """
    train_loss = [e['train_loss'] for e in history]
    val_loss = [e['val_loss'] for e in history]
    train_acc = [e['train_acc'] for e in history]
    val_acc = [e['val_acc'] for e in history]
    epochs = range(1, len(history) + 1) 
    
    fig, axs = plt.subplots(1, 2, figsize = [8, 4], layout = 'tight') 
    axs[0].set(xlabel = 'Epoch', ylabel = 'Loss')
    axs[1].set(xlabel = 'Epoch', ylabel = 'Accuracy')
    
    axs[0].plot(epochs, train_loss, color = plt.cm.viridis(0.),
                label = 'Train loss')
    axs[0].plot(epochs,   val_loss, color = plt.cm.viridis(0.33),
                label =   'Val loss')
    
    axs[1].plot(epochs, train_acc, color = plt.cm.viridis(0.),
                label = 'Train accuracy')
    axs[1].plot(epochs,   val_acc, color = plt.cm.viridis(0.33),
                label =   'Val accuracy')
    
    axs[0].legend(framealpha = 1)
    axs[1].legend(framealpha = 1)
    return fig, axs

################################################################################
################################## Evaluation ##################################
################################################################################

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate a model on a test dataset 

    Parameters:
    model (IMDbSentimentLSTM): model to evaluate 
    dataloader (DataLoader): test dataset loader 
    criterion: torch criterion for model evaluation 
    device (str): 'cpu' for CPU or 'cuda' for GPU
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item() * inputs.size(0)

            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy