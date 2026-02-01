import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class LearnablePositionalEncoding(nn.Module):
    """
    """

    def __init__(self, d_model, max_seq_len=128, dropout=0.0):
        """
        """

        super().__init__()
        self.lookup = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        return
    
    def forward(self, x):
        """
        """

        B, T, D = x.shape
        indices = torch.arange(T).to(device=x.device)
        embeddings = self.lookup(indices) # (T, D)
        y = x + embeddings # (B, T, D)
        y = self.dropout(y)

        return y

class MultiheadAttentionLayer(nn.Module):
    """
    """

    def __init__(self, d_model, n_heads=1, dropout=0.0):
        """
        """

        super().__init__()
        if d_model % n_heads != 0:
            raise Exception("Model width must be evenly divisible by the number of heads")
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, d_model)

        return
    
    def forward(self, query, key, value, attn_mask=None):
        """
        keywords
        --------
        query (B, T_src, D)
        key   (B, T_dst, D)
        value (B, T_dst, D)
        """

        #
        B, T_src, _ = query.shape
        _, T_dst, _ = key.shape


        # Project
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape to (B, T, n_heads, d_head)
        Q = Q.reshape(B, T_src, self.n_heads, self.d_head)
        K = K.reshape(B, T_dst, self.n_heads, self.d_head)
        V = V.reshape(B, T_dst, self.n_heads, self.d_head)

        # Transpose 1st and 2nd axes (swap time with head)
        Q = torch.transpose(Q, 1, 2) # (B, n_heads, T_src, d_head)
        K = torch.transpose(K, 1, 2) # (B, n_heads, T_dst, d_head)
        V = torch.transpose(V, 1, 2) # (B, n_heads, T_dst, d_head)

        # Transpose K
        K = torch.transpose(K, -2, -1) # (B, n_heads, d_head, T_dst)

        # TODO: Apply mask

        # Compute dot product
        dot_product = torch.matmul(Q, K) / np.sqrt(self.d_head) # (B, n_heads, T_src, T_dst)

        # Convert to probabilities
        scores = F.softmax(dot_product, dim=-1)

        # Apply dropout
        scores = self.dropout(scores)

        # Compute attention
        scores = scores @ V

        # Merge output from each head
        scores = scores.reshape(B, T_src, self.d_model)

        # Mix heads
        scores = self.head(scores)

        return scores
    
class SelfAttentionBlock(nn.Module):
    """
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        """

        super().__init__()
        self.attn = MultiheadAttentionLayer(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        return
    
    def forward(self, seq, mask):
        """
        """

        x = self.attn(seq, seq, seq)
        x = x + seq # Residual connection
        x = self.norm(x)
        x = self.dropout(x)

        return x

class FeeedForwardBlock(nn.Module):
    """
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        """

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        return
    
    def forward(self, seq):
        """
        """

        x = self.fc(seq)
        x = x + seq
        x = self.dropout(x)
        x = self.norm(x)

        return x
    
class EncoderLayer(nn.Module):
    """
    """

    def __init__(self, d_model, d_ff, n_heads, dropout):
        """
        """

        super().__init__()
        self.attn = SelfAttentionBlock(d_model, n_heads, dropout)
        self.ff = FeeedForwardBlock(d_model, d_ff, dropout)

        return
    
    def forward(self, seq, mask):
        """
        """

        x = self.attn(seq)
        x = self.ff(x)

        return x
    
class VisualTransformerClassifier(nn.Module):
    """
    """

    def __init__(self, patch_size, d_model, d_ff, n_heads, n_layers, n_classes, dropout=0.1, device=None):
        """
        """

        super().__init__()

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(
            in_features=self.patch_size * self.patch_size * 3,
            out_features=d_model
        )
        self.pe = LearnablePositionalEncoding(d_model, dropout=dropout)
        self.mlp = nn.Linear(d_model, n_classes)
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = list()
        for i_layer in range(n_layers):
            layer = EncoderLayer(d_model, d_ff, n_heads, dropout)
            self.layers.append(layer)

        #
        self.apply(self._init_weights)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.to(device)

        return
    
    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _patchify(self, images):
        """
        Patchify images

        Inputs
        ------
        images (N [or B], C, H, W)
        """

        unfold = nn.Unfold(self.patch_size, stride=self.patch_size).to(self.device)
        patches = unfold(images).permute(0, 2, 1) # (N [ or B], seq len, patch size * patch size * 3)

        return patches

    def forward(self, images):
        """
        """

        #
        B, C, H, W = images.size()

        # Patchify and embed images
        patches = self._patchify(images) # (B, T, patch size * patch size * 3)
        patches = self.patch_embedding(patches) # (B, T, d_model)

        # Concatenate class token with patches
        token = self.token.expand(B, -1, -1) # (B, 1, d_model)
        sequences = torch.cat([token, patches], dim=1)

        # Add positional encoding
        output = self.pe(sequences)

        # Run through encoder layers
        for layer in self.layers:
            output = layer(output) # (B, T, d_model)

        # Run through MLP
        output = output[:, 0] # Grab the class token
        output = self.mlp(output) 

        return output
    
def demo_with_cifar():
    """
    """

    return