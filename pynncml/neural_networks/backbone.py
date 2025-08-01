import torch
import torch.nn as nn
from pynncml import neural_networks
from pynncml.neural_networks.normalization import InputNormalization


class MetadataFCBlock(nn.Module):
    """
    A block of fully connected layers for processing metadata.
    
    :param input_size: Input dimension
    :param output_size: Output dimension  
    :param hidden_size: Hidden layer dimension (defaults to output_size)
    :param n_hidden: Number of hidden layers
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int | None = None, n_hidden: int = 0):
        super(MetadataFCBlock, self).__init__()
        
        if hidden_size is None:
            hidden_size = output_size
            
        layers = []
        
        # Input layer
        if n_hidden == 0:
            # Direct connection from input to output
            layers.append(nn.Linear(input_size, output_size))
        else:
            # Input to first hidden layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for _ in range(n_hidden - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            
            # Last hidden layer to output
            layers.append(nn.Linear(hidden_size, output_size))
        
        self.metadata_block = nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure input is float32 to match model parameters
        x = x.float()
        return self.metadata_block(x)


class Backbone(nn.Module):
    """
    The module is the backbone present in [1]

    :param n_layers: integer that state the number of recurrent layers.
    :param rnn_type: enum that define the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class tr.neural_networks.InputNormalizationConfig which hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number which define the alpha factor of time normalization layer.
    :param rnn_input_size: int that represent the dynamic input size.
    :param rnn_n_features: int that represent the dynamic feature size.
    :param metadata_input_size: int that represent the metadata input size.
    :param metadata_n_features: int that represent the metadata feature size.
    :param metadata_n_hidden: int that represent the number of hidden layers in the metadata fc block.
    :param metadata_feature_mask: Optional tensor mask to control which metadata features are used. 
                                 Should be a boolean tensor of shape [metadata_input_size] or None (uses all features).
    :param freeze_rnn: boolean that controls whether to freeze the RNN parameters (default: False).
                       When True, RNN parameters are frozen but metadata block remains trainable.
    """

    def __init__(self, n_layers: int, rnn_type: neural_networks.RNNType,
                 normalization_cfg: neural_networks.InputNormalizationConfig,
                 enable_tn: bool,
                 tn_alpha: float,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 metadata_n_hidden: int = 0, # number of hidden layers in the metadata fc block
                 metadata_feature_mask = None,
                 freeze_rnn: bool = False,
                 ):

        super(Backbone, self).__init__()
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.metadata_n_features = metadata_n_features
        self.rnn_n_features = rnn_n_features
        self.metadata_n_hidden = metadata_n_hidden # number of hidden layers in the metadata fc block
        self.freeze_rnn = freeze_rnn
        
        # Handle metadata feature mask
        if metadata_feature_mask is not None:
            if not isinstance(metadata_feature_mask, torch.Tensor):
                raise ValueError("metadata_feature_mask must be a torch.Tensor or None")
            
            metadata_feature_mask = metadata_feature_mask.bool()
            
            if metadata_feature_mask.shape[0] != metadata_input_size:
                raise ValueError(f"metadata_feature_mask length ({metadata_feature_mask.shape[0]}) must match metadata_input_size ({metadata_input_size})")
            
            self.register_buffer('metadata_feature_mask', metadata_feature_mask)
        else:
            # If no mask provided, use all features (mask of all ones)
            self.register_buffer('metadata_feature_mask', torch.ones(metadata_input_size, dtype=torch.bool))
        
        # Model Layers
        if rnn_type == neural_networks.RNNType.GRU:
            self.rnn = nn.GRU(rnn_input_size, rnn_n_features,
                              bidirectional=False, num_layers=n_layers,
                              batch_first=True)
        elif rnn_type == neural_networks.RNNType.LSTM:
            self.rnn = nn.LSTM(rnn_input_size, rnn_n_features,
                               bidirectional=False, num_layers=n_layers,
                               batch_first=True)
        else:
            raise Exception('Unknown RNN type')
        
        # Freeze RNN parameters if requested
        if self.freeze_rnn:
            for param in self.rnn.parameters():
                param.requires_grad = False
        
        self.enable_tn = enable_tn
        if enable_tn:
            self.tn = neural_networks.TimeNormalization(alpha=tn_alpha, num_features=rnn_n_features)
        
        # Replace single fc_meta with metadata processing block
        self.metadata_block = MetadataFCBlock(
            input_size=metadata_input_size,
            output_size=metadata_n_features,
            hidden_size=metadata_n_features,  # Use metadata_n_features as hidden size
            n_hidden=metadata_n_hidden
        )
        
        # self.fc_meta = nn.Linear(metadata_input_size, metadata_n_features)
        self.normalization = InputNormalization(normalization_cfg)
        self.relu = nn.ReLU()

    def total_n_features(self) -> int:
        """
        This function return the total number of feature generated by the backbone

        :return: integer number state the total number of feature.
        """
        return self.metadata_n_features + self.rnn_n_features

    def forward(self, data: torch.Tensor, metadata: torch.Tensor, state: torch.Tensor):
        """
        This is the module forward function

        :param data: A tensor of the dynamic data of shape :math:`[N_b,N_s,N_i^d]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_i^d` is the dynamic input size.
        :param metadata:  A tensor of the metadata of shape :math:`[N_b,N_i^m]` where :math:`N_b` is the batch size,
                          and :math:`N_i^m` is the metadata input size.
        :param state: A tensor that represent the state of shape
        :return: Two Tensors, the first tensor if the feature tensor of size :math:`[N_b,N_s,N_f]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and :math:`N_f` is the number of feature.
                    The second tensor is the state tensor.
        """
        input_tensor, input_meta_tensor = self.normalization(data, metadata)
        
        # Apply metadata feature mask
        # Expand mask to match batch dimension and apply to input_meta_tensor
        mask_expanded = self.metadata_feature_mask.unsqueeze(0).expand(input_meta_tensor.size(0), -1)
        masked_meta_tensor = input_meta_tensor * mask_expanded.float()
        
        if self.enable_tn:  # split hidden state for RE
            hidden_tn = state[1]
            hidden_rnn = state[0]
        else:
            hidden_rnn = state
            hidden_tn = None
        
        # Use the metadata block instead of single fc_meta layer
        output_meta = self.relu(self.metadata_block(masked_meta_tensor))
        # output_meta = self.relu(self.fc_meta(input_meta_tensor))
        
        output, hidden_rnn = self.rnn(input_tensor, hidden_rnn)
        output = output.contiguous()
        output_meta = output_meta.view(output_meta.size()[0], 1, output_meta.size()[1]).repeat(1, output.size()[1], 1)
        ##############################################################################
        if self.enable_tn:  # run TimeNormalization over rnn output and update the state of Backbone
            output_new, hidden_tn = self.tn(output, hidden_tn)
            hidden = (hidden_rnn, hidden_tn)
        else:  # pass rnn state and output
            output_new = output
            hidden = hidden_rnn
        ##############################################################################
        features = torch.cat([output_new, output_meta], dim=-1)

        return features, hidden

    def _base_init(self, batch_size: int = 1) -> torch.Tensor:
        """
        This function generate the initial state for the recurrent layer.

        :param batch_size: int represent the batch size.
        """
        return torch.zeros(self.n_layers, batch_size, self.rnn_n_features,
                           device=self.metadata_block.metadata_block[0].weight.device)  # create inital state for rnn layer only

    def init_state(self, batch_size: int = 1):
        """
        This function generate the initial state of the Module. This include both the recurrent layers state and Time Normalization state

        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        if self.rnn_type == neural_networks.RNNType.GRU:
            state = self._base_init(batch_size)
        else:
            state = (self._base_init(batch_size), self._base_init(batch_size))

        if self.enable_tn:  # if TimeNormalization is enable then update init state
            state = (state, self.tn.init_state(self.metadata_block.metadata_block[0].weight.device.type, batch_size=batch_size))
        return state
