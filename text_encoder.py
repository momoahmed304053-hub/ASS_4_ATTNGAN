"""
Text Encoder Module for Res_AttnGAN
Encodes text descriptions into embeddings
"""

import torch
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU


class TextEncoder(nn.Module):
    """
    Text encoder using BiLSTM to encode text descriptions
    
    Args:
        vocab_size: Size of vocabulary
        word_dim: Dimension of word embeddings (default: 300)
        sent_dim: Dimension of sentence embeddings (default: 256)
    """
    
    def __init__(self, vocab_size=5000, word_dim=300, sent_dim=256):
        super(TextEncoder, self).__init__()
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, word_dim)
        
        # BiLSTM for word-level features
        self.lstm = nn.LSTM(
            input_size=word_dim,
            hidden_size=sent_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection layer for word embeddings
        self.word_projection = nn.Linear(sent_dim, sent_dim)
        
    def forward(self, captions, lengths):
        """
        Forward pass for text encoder
        
        Args:
            captions: (batch_size, max_length) - token indices
            lengths: (batch_size,) - actual lengths of captions
            
        Returns:
            word_embs: (batch_size, max_length, sent_dim) - word-level embeddings
            sent_embs: (batch_size, sent_dim) - sentence-level embeddings
        """
        # Embed words
        word_embs = self.word_embedding(captions)  # (batch_size, max_length, word_dim)
        
        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            word_embs, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Apply BiLSTM
        lstm_out, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        word_features, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # (batch_size, max_length, sent_dim)
        
        # Project word features
        word_embs = self.word_projection(word_features)
        
        # Get sentence embedding from final hidden states
        # Concatenate forward and backward hidden states
        sent_embs = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch_size, sent_dim)
        
        return word_embs, sent_embs


class RobertaTextEncoder(nn.Module):
    """
    Text encoder using pre-trained Roberta model
    More advanced option for better semantic understanding
    """
    
    def __init__(self, sent_dim=256):
        super(RobertaTextEncoder, self).__init__()
        try:
            from transformers import RobertaModel, RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta = RobertaModel.from_pretrained('roberta-base')
            self.sent_dim = sent_dim
            self.projection = nn.Linear(768, sent_dim)
        except ImportError:
            print("Warning: transformers not installed. Use TextEncoder instead.")
            
    def forward(self, captions, device):
        """
        Forward pass for Roberta encoder
        
        Args:
            captions: list of text strings
            device: torch device
            
        Returns:
            word_embs: word-level embeddings
            sent_embs: sentence-level embeddings
        """
        try:
            # Tokenize
            encoded = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Get embeddings from Roberta
            outputs = self.roberta(**encoded)
            
            # last_hidden_state: (batch_size, seq_len, 768)
            word_features = outputs.last_hidden_state
            
            # Sentence embedding: use [CLS] token (first token)
            sent_embs = word_features[:, 0, :]  # (batch_size, 768)
            
            # Project to target dimension
            word_embs = self.projection(word_features)  # (batch_size, seq_len, sent_dim)
            sent_embs = self.projection(sent_embs)  # (batch_size, sent_dim)
            
            return word_embs, sent_embs
        except:
            raise NotImplementedError("Roberta encoder requires transformers library")
