from transformers import (
    AutoModel,
    T5EncoderModel,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ESLM(nn.Module):
    """
    Employing contextual language models for entity summarization tasks
    """
    def __init__(self, model_name, model_base, mlp_hidden_dim=512):
        """
        Model initialization
        
        Attributes:
            - model_name (str): the name of contextual language model in lower case 
                (e.g., 't5', 'bert', 'ernie').
            - model_base (str): the parameter specifies the exact pre-trained model variant to be loaded 
                (e.g., 't5-base', 'bert-base-uncased') related to the model name.
            - mlp_hidden_dim (int): the size of the hidden layers within the Multi-Layer Perceptron (MLP) part of the model.
        """
        super(ESLM, self).__init__()
        self.model_name = model_name
        self.model_base = model_base
        if self.model_name=="t5":
            self.lm_encoder = T5EncoderModel.from_pretrained(model_base)
            self.feat_dim = self.lm_encoder.config.d_model
        else:
            self.lm_encoder = AutoModel.from_pretrained(model_base)
            self.feat_dim = list(self.lm_encoder.modules())[-2].out_features
            
        self.attention = nn.Linear(self.feat_dim, 1)
        self.regression = nn.Linear(self.feat_dim, 1)  # Output layer for regression
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # Output layer for regression
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Attributes:
            - input_ids (tensor): sequences of integers representing tokens mapped from a vocabulary
            - attention_mask (tensor): the tensors comprise of 1 and 0 to helps the model to distinguish 
                between meaningful data and padding data
        """
        encoder_output = self.lm_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        attn_weights = F.softmax(self.attention(encoder_output), dim=-1)
        combined_output = attn_weights * encoder_output
        
        # Pass through MLP
        regression_output = self.mlp(combined_output)
        
        # Averaging across the sequence
        regression_output = regression_output.mean(dim=1)  # This averages the output across the sequence

        # Apply activation 
        # For outputs bounded between 0 and 1
        regression_output = F.softmax(regression_output, dim=0) # use dim=0 due to not implemented data batches

        return regression_output

class ESLMKGE(nn.Module):
    """
    Implementation ESLM Enrichment by augmenting Knowledge Graph Embeddings(KGEs)
    """
    def __init__(self, model_name, model_base, kg_embedding_dim=1200, mlp_hidden_dim=512):
        """
        Model initialization
        
        Attributes:
            - model_name (str): the name of contextual language model in lower case 
                (e.g., 't5', 'bert', 'ernie').
            - model_base (str): the parameter specifies the exact pre-trained model variant to be loaded 
                (e.g., 't5-base', 'bert-base-uncased') related to the model name.
            - kg_embedding_dim (int): the size of the embeddings used for the knowledge graph components
            - mlp_hidden_dim (int): the size of the hidden layers within the Multi-Layer Perceptron (MLP) part of the model.
        """
        super(ESLMKGE, self).__init__()
        if model_name == "t5":
            self.lm_encoder = T5EncoderModel.from_pretrained(model_base)
            self.feat_dim = self.lm_encoder.config.d_model
        else:
            self.lm_encoder = AutoModel.from_pretrained(model_base)
            self.feat_dim = list(self.lm_encoder.modules())[-2].out_features

        # Attention layer
        self.attention = nn.Linear(self.feat_dim + kg_embedding_dim, 1)
        
        # Refression layer
        self.regression = nn.Linear(self.feat_dim + kg_embedding_dim, 1)  # Output layer for regression
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim + kg_embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # Output layer for regression
        )

    def forward(self, input_ids, attention_mask, kg_embeddings):
        """
        Forward pass
        
        Attributes:
            - input_ids (tensor): sequences of integers representing tokens mapped from a vocabulary
            - attention_mask (tensor): the tensors comprise of 1 and 0 to helps the model to distinguish 
                between meaningful data and padding data
            - kg_embeddings (tensor): the tensors represent KGEs, where each embedding is likely 
                a vectorized representation of a triple (e.g., subject, predicate, object) from 
                a knowledge graph
        """
        encoder_output = self.lm_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Expand KG Embeddings
        kg_embeddings_expanded = kg_embeddings.expand(-1, encoder_output.size(1), -1)

        # Combine with lm encoder output
        combined_embeddings = torch.cat([encoder_output, kg_embeddings_expanded], dim=-1)
        pooled_output = combined_embeddings.mean(dim=1)
        
        # Apply attention mechanism
        attn_weights = F.softmax(self.attention(pooled_output), dim=-1)
        combined_output = attn_weights * pooled_output
        
        # Pass through MLP
        regression_output = self.mlp(combined_output)
        
        # Apply activation 
        # For outputs bounded between 0 and 1
        regression_output = F.softmax(regression_output, dim=0) # use dim=0 due to not implemented data batches

        return regression_output
