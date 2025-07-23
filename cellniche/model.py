import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, LeakyReLU, PReLU
from torch_geometric.nn import SAGEConv
from typing import Optional, List, Tuple



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for graph embeddings using a positive pair mask.

    Args:
        temperature (float): Initial temperature parameter for scaling similarities.
        scale_by_temperature (bool): If True, scales the loss by temperature.
        use_weight (bool): If True, applies weighting to positive pairs.
        pos_weight_strategy (str): Strategy for positive pair weights: 'none', 'inverse_freq', 'inverse_sim'.
        neg_weight_strategy (str): Strategy for negative pair weights: 'none', 'inverse_sim'.
        pos_generation_strategy (str): Strategy influencing positive weight generation.
    """
    def __init__(
        self,
        temperature: float = 0.5,
        scale_by_temperature: bool = False,
        use_weight: bool = True,
        pos_weight_strategy: str = 'inverse_sim',
        neg_weight_strategy: str = 'inverse_sim',
        pos_generation_strategy: str = 'inverse_sim',
    ):
        super().__init__()
        # Register temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.scale_by_temperature = scale_by_temperature
        self.use_weight = use_weight
        self.pos_weight_strategy = pos_weight_strategy
        self.neg_weight_strategy = neg_weight_strategy
        self.pos_generation_strategy = pos_generation_strategy

    def forward(self, embeddings: torch.Tensor, pos_mask: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings (Tensor): Node embeddings of shape (batch, dim).
            pos_mask (SparseTensor): Sparse mask indicating positive pairs.

        Returns:
            Tensor: Scalar loss value.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Normalize embeddings and scale by temperature
        z = F.normalize(embeddings, p=2, dim=1)
        temp = F.softplus(self.temperature)
        sim_matrix = torch.matmul(z, z.t()) / temp
        # For numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

        # Mask out self-comparisons
        identity = torch.eye(batch_size, device=device).bool()

        # Construct dense positive and negative masks
        pos_dense = torch.zeros_like(sim_matrix, dtype=torch.bool)
        # pos_dense[pos_mask.indices()[0], pos_mask.indices()[1]] = True
        pos_dense[pos_mask.storage.row(), pos_mask.storage.col()] = True
        neg_dense = ~pos_dense & ~identity

        # Compute log probabilities
        log_probs = F.log_softmax(sim_matrix, dim=1)
        pos_log = log_probs[pos_dense]

        if self.use_weight:
            # Compute positive weights
            values = pos_mask.values().float()
            if self.pos_weight_strategy == 'inverse_freq':
                pos_w = 1.0 / (values + 1e-8)
            elif self.pos_weight_strategy == 'inverse_sim':
                pos_w = 1.0 - values
            else:
                pos_w = torch.ones_like(values)
            # Normalize weights to [0,1]
            pos_w = (pos_w - pos_w.min()) / (pos_w.max() - pos_w.min() + 1e-8)

            # Weighted positive loss
            loss = - (pos_log * pos_w).mean()
        else:
            # Standard contrastive loss
            loss = - pos_log.mean()

        return loss


class Encoder(nn.Module):
    """
    Graph Encoder with multiple SAGEConv layers.

    Args:
        in_channels (int): Dimensionality of input node features.
        hidden_channels (List[int]): Hidden sizes for each GNN layer.
        conv_layer (callable): Graph convolution class (default: SAGEConv).
        dropout (float): Dropout probability between layers.
        negative_slope (float): Negative slope for LeakyReLU.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        conv_layer=SAGEConv,
        dropout: float = 0.0,
        negative_slope: float = 0.5,
    ):
        super().__init__()
        # Build graph convolutional layers
        self.convs = nn.ModuleList(
            conv_layer(
                in_channels if i == 0 else hidden_channels[i - 1],
                hidden_channels[i]
            )
            for i in range(len(hidden_channels))
        )
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize all convolutional layers."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        adjs: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int,int]]]] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN layers.

        Args:
            x (Tensor): Node feature matrix.
            adjs (list, optional): List of (edge_index, e_id, size) for sampling.
            edge_index (Tensor, optional): Full-graph edge index if adjs is None.

        Returns:
            h (Tensor): Output node embeddings.
            target (Tensor): Features of target nodes for reconstruction.
        """
        if adjs is None:
            # Use full-graph edge index
            assert edge_index is not None, "edge_index required if adjs is None"
            layers = [(edge_index, None, (x.size(0), x.size(0)))]
            target = x
        else:
            layers = adjs
            # Target nodes correspond to first N' rows
            target = x[: layers[-1][2][1]]

        for i, (e_idx, _, size) in enumerate(layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_target = x[: size[1]]
            x = self.convs[i]((x, x_target), e_idx)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)

        return x, target


class Model(nn.Module):
    """
    Full Graph Contrastive Model with optional projection and reconstruction.

    Args:
        encoder (Encoder): Graph encoder instance.
        decoder_hidden (List[int], optional): Hidden sizes for decoder MLP.
        project_hidden (List[int], optional): Hidden sizes for projection head.
        tau (float): Temperature parameter for contrastive loss.
        use_weight (bool): Use weighted contrastive loss.
        pos_weight_strategy (str): Positive weight strategy.
        neg_weight_strategy (str): Negative weight strategy.
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder_hidden: Optional[List[int]] = None,
        project_hidden: Optional[List[int]] = None,
        tau: float = 0.5,
        use_weight: bool = True,
        pos_weight_strategy: str = 'none',
        neg_weight_strategy: str = 'none',
    ):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.convs[-1].out_channels

        # Projection head for contrastive learning
        if project_hidden:
            layers = []
            in_dim = embed_dim
            for dim in project_hidden:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(PReLU())
                in_dim = dim
            self.project = nn.Sequential(*layers)
        else:
            self.project = None

        # Decoder for reconstruction (optional)
        if decoder_hidden:
            layers = []
            in_dim = embed_dim
            for dim in decoder_hidden:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(BatchNorm1d(dim))
                layers.append(LeakyReLU())
                in_dim = dim
            layers.append(nn.Linear(in_dim, encoder.convs[0].in_channels))
            self.decoder = nn.Sequential(*layers)
            self.recon_loss_fn = nn.MSELoss()
        else:
            self.decoder = None
            self.recon_loss_fn = None

        # Contrastive loss function
        self.loss_fn = ContrastiveLoss(
            temperature=tau,
            use_weight=use_weight,
            pos_weight_strategy=pos_weight_strategy,
            neg_weight_strategy=neg_weight_strategy,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjs: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]]] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returns:
          - h: encoder output embeddings
          - z: projected embeddings (for contrastive loss)
          - rec: reconstructed features (for decoder)
          - target: original target features
        """
        h, target = self.encoder(x, adjs, edge_index)
        z = self.project(h) if self.project else h
        rec = self.decoder(h) if self.decoder else h
        
        return h, z, rec, target

    def compute_loss(
        self,
        z: torch.Tensor,
        pos_mask: torch.sparse.FloatTensor,
        proj_z: torch.Tensor,
        rec: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive and reconstruction losses.

        Args:
            z (Tensor): Projected embeddings.
            pos_mask (SparseTensor): Mask of positive pairs.
            rec (Tensor, optional): Reconstructed features.
            target (Tensor, optional): Original features for reconstruction.

        Returns:
            contrast_loss, recon_loss (Tensors)
        """
        # Contrastive loss
        if self.project:
            contrast_loss = self.loss_fn(proj_z, pos_mask)
        else:
            contrast_loss = self.loss_fn(z, pos_mask)

        # Reconstruction loss
        if self.decoder:
            if target is None:
                raise ValueError("Target features required for reconstruction loss.")
            recon_loss = self.recon_loss_fn(rec, target)
        else:
            recon_loss = torch.tensor(0.0, device=z.device)

        return contrast_loss, recon_loss
