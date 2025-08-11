import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This is a standard implementation of positional encoding, using sine and
    cosine functions of different frequencies.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class WhoFiTransformer(nn.Module):
    """
    The main Transformer-based model for WhoFi.
    This model processes flattened CSI data to generate a biometric signature.
    """
    def __init__(self, input_dim=342, d_model=128, nhead=8, num_encoder_layers=1,
                 dim_feedforward=512, dropout=0.1, signature_dim=128):
        super(WhoFiTransformer, self).__init__()

        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # Expects (seq_len, batch, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.signature_fc = nn.Linear(d_model, signature_dim)
        self.d_model = d_model

    def forward(self, src):
        """
        Forward pass of the WhoFiTransformer model.

        Args:
            src (torch.Tensor): The source input tensor.
                                Shape: (batch_size, seq_len, input_dim), e.g., (B, P, 342).

        Returns:
            torch.Tensor: The output signature tensor. Shape: (batch_size, signature_dim).
        """
        # Project input to d_model
        src = self.input_fc(src) * math.sqrt(self.d_model)

        # PyTorch Transformer expects (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(src)

        # Take the output of the first token (analogous to a CLS token)
        # as the representation of the whole sequence.
        output = output[0, :, :]

        # Generate the signature
        signature = self.signature_fc(output)

        # L2-normalize the signature
        signature = nn.functional.normalize(signature, p=2, dim=1)

        return signature
