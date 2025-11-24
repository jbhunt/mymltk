from torch import nn
import torch
import math
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

# Constants
FLOAT_DTYPE = torch.float32
INTEGER_DTYPE = torch.long

# TODO: Integrate the vocabulary with the Dataset class itself
class PhonemeVocabulary():
    """
    """

    PAD = 0
    BOS = 1
    EOS = 2
    map = {
        0: "<PAD>",
        1: "<BOS>",
        2: "<EOS",
        3: "<BLANK>",
        4: "AA",
        5: "AE",
        6: "AH",
        7: "AO",
        8: "AW",
        9: 'AY',
        10: 'B',
        11: 'CH',
        12: 'D',
        13: 'DH',
        14: 'EH',
        15: 'ER',
        16: 'EY',
        17: 'F',
        18: 'G',
        19: 'HH',
        20: 'IH', 
        21: 'IY',
        22: 'JH',
        23: 'K',
        24: 'L',
        25: 'M',
        26: 'N',
        27: 'NG',
        28: 'OW',
        29: 'OY',
        30: 'P',
        31: 'R',
        32: 'S',
        33: 'SH',
        34: 'T',
        35: 'TH',
        36: 'UH',
        37: 'UW',
        38: 'V',
        39: 'W',
        40: 'Y',
        42: 'Z',
        43: 'ZH',
        44: '<SILENCE>'
    }

    def __init__(self):
        """
        """

        return
    
    def insert_special_tokens(self, in_seq, padding_token=0):
        """
        """

        in_seq = np.array(in_seq)
        out_seq = np.copy(in_seq)
        mask = np.array(out_seq) != padding_token
        out_seq[mask] += 2
        i = np.where(in_seq ==  0)[0][0]
        out_seq = np.insert(out_seq, i, self.EOS)
        out_seq = np.concatenate([
            np.array([self.BOS]),
            out_seq
        ])

        return out_seq
    
    def decode(self, in_seq):
        """
        """

        out_seq = list()
        for el in in_seq:
            # if el == self.BOS:
            #     continue
            # elif el == self.EOS:
            #     break
            token = self.map[el]
            out_seq.append(token)

        return np.array(out_seq)
    
    @property
    def size(self):
        return len(self.map)

def make_key_padding_mask(batch, seq_lens):
    """
    """

    B, T, N = batch.size()
    mask = torch.full([B, T], True)
    for i_x, x in enumerate(batch):
        seq_len = seq_lens[i_x]
        mask[i_x, :seq_len] = False

    return mask

def make_causal_mask(batch):
    """
    """

    B, T, N = batch.size()
    mask = torch.triu(
        torch.full((T, T), float("-inf")),
        diagonal=1,
    )
    return mask

class PositionalEncoding(nn.Module):
    """
    """

    def __init__(self, seq_len, d_model, base=10000):
        """
        """

        super().__init__()
        pe = torch.zeros(seq_len, d_model, dtype=FLOAT_DTYPE)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        return
    
    def forward(self, X):
        """
        """

        B, T, N = X.size()

        return self.pe[:T, :].unsqueeze(0) # B x T x D
    
class FrontendModule(nn.Module):
    """
    Frontend module that handles positional encoding and session embedding
    """

    def __init__(
        self,
        d_sessions,
        n_channels,
        d_model,
        seq_len,
        base=10000
        ):
        """
        """

        #
        super().__init__()

        self.se = nn.Linear(n_channels, d_sessions, dtype=FLOAT_DTYPE)
        self.fc = nn.Linear(n_channels + d_sessions, d_model)
        self.pe = PositionalEncoding(seq_len, d_model, base=base)

        return
    
    def forward(self, X, z):
        """
        """

        B, T, N = X.size()
        X_1 = self.se(z).unsqueeze(1).expand(-1, T, -1)
        X_2 = torch.cat([X, X_1], dim=-1) # B x T x (N + d_session)
        X_3 = self.fc(X_2) # B x T x D
        X_4 = X_3 + self.pe(X)

        return X_4
    
class BrainToTextEncoderDecoderTransformer(nn.Module):
    """
    """

    def __init__(
        self,
        vocab_size=44, # N possible phoneme tokens
        d_sessions=16,
        d_model=128,
        src_seq_len=500,
        tgt_seq_len=501,
        n_channels=256,
        n_heads=8,
        n_layers=4
        ):
        """
        """

        super().__init__()

        #
        self.frontend = FrontendModule(
            d_sessions,
            n_channels,
            d_model,
            src_seq_len
        )

        #
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead=n_heads,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # Embed target phoneme sequence and encode sequence order
        self.pho = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(
            tgt_seq_len,
            d_model
        )

        #
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                batch_first=True
            ),
            num_layers=n_layers
        )

        #
        self.clf = nn.Linear(in_features=d_model, out_features=vocab_size)

        return
    
    def forward(self, X, y, z, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        """

        # Check data types
        if X.dtype != FLOAT_DTYPE:
            X = X.to(FLOAT_DTYPE)
        if z.dtype != FLOAT_DTYPE:
            z = z.to(FLOAT_DTYPE)
        if y.dtype != torch.long:
            y = y.long()

        #
        B, T_src, N_src = X.size()

        # Encoder block
        X_1 = self.frontend(X, z)
        X_2 = self.encoder(
            X_1,
            src_key_padding_mask=memory_key_padding_mask
        )

        # Embed phoneme sequence and add positional encoding
        X_3 = self.pho(y)
        X_4 = X_3 + self.pe(X_3)

        # Decoder block
        tgt_mask = make_causal_mask(X_4)
        X_5 = self.decoder(
            tgt=X_4,
            memory=X_2,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, # [B, T_tgt] or None
            memory_key_padding_mask=memory_key_padding_mask, # [B, T_src] or None
        ) 

        #
        logits = self.clf(X_5)

        return logits
    
class BrainToTextDecoder():
    """
    """

    def __init__(self, src_seq_len=500, tgt_seq_len=501, max_iter=100, **kwargs):
        """
        """

        #
        self.vocab = PhonemeVocabulary()
        self.model = BrainToTextEncoderDecoderTransformer(
            src_seq_len=src_seq_len,
            tgt_seq_len=tgt_seq_len,
            vocab_size=self.vocab.size,
            **kwargs
        )
        self.max_iter = max_iter
        self.loss = None

        return
    
    def fit(self, ds, batch_size=10):
        """
        """

        #
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        n_batches = len(ds)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss = np.full(self.max_iter, np.nan)

        #
        self.model.train()
        for i_epoch in range(self.max_iter):
            batch_loss = 0.0    
            for i_batch, (X_batch, y_batch, z_batch, src_seq_lens) in enumerate(loader):

                #
                y_batch = np.array([self.vocab.insert_special_tokens(yi) for yi in y_batch]) # Insert pad, bos, and eos tokens
                y_batch = torch.tensor(y_batch, dtype=FLOAT_DTYPE)
                y_inputs = y_batch[:, :-1]
                y_outputs = y_batch[:, 1:]

                #
                src_key_padding_mask = make_key_padding_mask(X_batch, src_seq_lens)
                tgt_seq_lens = list()
                tgt_key_padding_mask = None # TODO: Make a key padding mask for the target sequence (probably using y_inputs)

                #
                logits = self.model(
                    X_batch,
                    y_inputs,
                    z_batch,
                    src_key_padding_mask
                )

                #
                B, T, V = logits.size()
                loss = loss_fn(logits.view(-1, V), y_outputs.ravel().long())
                batch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            #
            batch_loss /= n_batches
            self.loss[i_epoch] = batch_loss

            if i_epoch + 1 == 3:
                break
        
        import pdb; pdb.set_trace()

        return
    
    def predict(self, X, z, src_seq_lens, max_tgt_seq_len=500):
        """
        """

        # Check data types
        if X.dtype != FLOAT_DTYPE:
            X = torch.tensor(X, dtype=FLOAT_DTYPE)
        if z.dtype != FLOAT_DTYPE:
            z = torch.tensor(z, dtype=FLOAT_DTYPE)

        # 
        src_key_padding_mask = make_key_padding_mask(X, src_seq_lens)
        B, T_src, N_src = X.shape
        y = torch.full(
            (B, 1),
            self.vocab.BOS,
            dtype=torch.long,
            device="cpu"
        )

        #
        self.model.eval()
        with torch.no_grad():
            for t in range(max_tgt_seq_len):
                all_logits = self.model(
                    X,
                    y,
                    z,
                    memory_key_padding_mask=src_key_padding_mask
                )
                next_token_logits = all_logits[:, -1, :] # B x V
                next_token = torch.argmax(next_token_logits, dim=1) # B x 1
                y = torch.cat([y, next_token.unsqueeze(1)], dim=1)

                # Check if all sequences produces an EOS token
                finished = torch.any(y == self.vocab.EOS, dim=1).all()
                if finished == True:
                    break

        #
        for i_seq, tgt_seq in enumerate(y):
            indices = torch.where(tgt_seq == self.vocab.EOS)[0]
            if len(indices) == 0:
                continue
            index = indices[0]
            y[i_seq, index + 1:] = self.vocab.PAD
        
        y = y.numpy()
        return y