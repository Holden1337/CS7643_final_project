import torch
import torch.nn as nn


class CTTS(nn.Module):
    """
    Class for our CTTS model
    """
    def __init__(self, device, kernel_size=16, stride=8, padding=0, d_model=128, nhead=4, enc_layers=4, dim_feedforward=512, dropout=0.3):
        super().__init__()
        # params
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.d_model = d_model
        self.nhead = nhead
        self.enc_layers = enc_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = device


        # model layers
        self.cnn_layer = nn.Conv1d(device=self.device, in_channels=1, out_channels=1, kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding)
        

        # TODO: We apparently need a positional enocoding layer and we will add the output of this to the output of the cnn
        # kind of like the transformer project but instead of two embedding layers (word and position) we only need one since
        # the cnn is doing the "token" embedding
        # reason i have the TODO is that I'm not sure about the dimensions
        self.pos_embed_layer = nn.Embedding(num_embeddings=80, embedding_dim=128, device=self.device)

        # transformer layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                                                        dropout=self.dropout, device=self.device)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        # final output layer
        #TODO: might want to experiment with more linear layers
        # in the paper it just says "MLP" in the diagram, they don't say how many
        # layers they used.
        self.final_layer = nn.Linear(in_features=self.d_model, out_features=3)

    
    def forward(self, x):
        """
        forward pass of CTTS model.
        Should in theory be:
        1. Pass time series (x) through CNN layer to create token embedding
        2. Pass token embedding through positional embedding layer to create final embed
        3. Pass final embed through transformer encoder
        4. Take outputs from transformer encoder and pass those though MLP layer
        5. Return outputs from MLP layer

        but we may have to trouble shoot

        :param x: torch tensor of size (figure this out later) time series input batch
        :return outputs: torch tensor of size (figure this out later) output of CTTS model
        """

        #token_embed = self.cnn_layer(x)
        # figure out positional embedding here
        #pos_embed = self.pos_embed_layer(some_input)
        #embedding = torch.add(token_embed, pos_embed)

        # transformer_output = self.transformer_encoder(embedding)

        #mlp_output = self.final_layer(transformer_output)

        # return mlp_output