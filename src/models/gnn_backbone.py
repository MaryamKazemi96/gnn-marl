"""
GNN backbone implementations for ego-graph encoding.
Supports both dummy MLP and GraphSAGE-based encoders.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops


class DummyBackbone(nn.Module):
    """Simple MLP baseline (no graph structure, node features only)."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n, in_dim] -> [n, hidden]"""
        return self.net(x)


class EdgeSAGEConv(MessagePassing):
    """
    GraphSAGE convolution with edge attributes.
    Concatenates sender, receiver, and edge features before MLP.
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [n, in_dim] node features
            edge_index: [2, e] edge indices
            edge_attr: [e, edge_dim] edge attributes
        Returns:
            [n, out_dim] aggregated node features
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(
        self, 
        x_i: torch.Tensor, 
        x_j: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute message from sender j to receiver i."""
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class EgoGraphEncoder(nn.Module):
    """
    GraphSAGE encoder for batched ego-graphs.
    Supports optional edge attributes (route/travel features).
    """
    def __init__(
        self, 
        in_dim: int, 
        hidden: int, 
        layers: int = 2, 
        edge_dim: int = 0
    ):
        """
        Args:
            in_dim: Node feature dimension
            hidden: Hidden layer dimension
            layers: Number of GNN layers
            edge_dim: Edge feature dimension (0 if no edge features)
        """
        super().__init__()
        self.edge_dim = int(edge_dim)
        
        # Build GNN layers
        if self.edge_dim > 0:
            # Use EdgeSAGEConv if edge features present
            gs = [EdgeSAGEConv(in_dim, hidden, self.edge_dim)]
            for _ in range(layers - 1):
                gs.append(EdgeSAGEConv(hidden, hidden, self.edge_dim))
        else:
            # Standard SAGEConv if no edge features
            gs = [SAGEConv(in_dim, hidden)]
            for _ in range(layers - 1):
                gs.append(SAGEConv(hidden, hidden))
        
        self.gnn = nn.ModuleList(gs)
        self.act = nn.ReLU()
    
    def forward(
        self,
        x_list: List[torch.Tensor],
        edge_index_list: List[torch.Tensor],
        edge_attr_list: Optional[List[torch.Tensor]] = None,
        _batch_list=None,  # Unused, for API compatibility
    ):
        """
        Encode a batch of ego-graphs into node embeddings.
        
        Args:
            x_list: List of [n_i, in_dim] node feature tensors per graph
            edge_index_list: List of [2, e_i] edge index tensors per graph
            edge_attr_list: Optional list of [e_i, edge_dim] edge attribute tensors
            _batch_list: Unused (for compatibility with other encoders)
        
        Returns:
            h: [sum_i n_i, hidden] combined node embeddings
            batch: PyG Batch object with .batch mapping nodes -> graph ids
        """
        # Create PyG Data objects
        datas: List[Data] = []
        if edge_attr_list is None:
            for x, ei in zip(x_list, edge_index_list):
                datas.append(Data(x=x, edge_index=ei))
        else:
            for x, ei, ea in zip(x_list, edge_index_list, edge_attr_list):
                datas.append(Data(x=x, edge_index=ei, edge_attr=ea))
        
        # Batch all graphs together
        batch: Batch = Batch.from_data_list(datas)  # type: ignore
        
        # Add self-loops (helps isolated nodes and graphs with no edges)
        if self.edge_dim > 0:
            edge_index, edge_attr = add_self_loops(
                batch.edge_index,
                batch.edge_attr,
                num_nodes=batch.x.size(0),
                fill_value=0.0,
            )
        else:
            edge_index, edge_attr = add_self_loops(
                batch.edge_index, 
                num_nodes=batch.x.size(0)
            )  # type: ignore
        
        # Forward through GNN layers
        h = batch.x  # type: ignore
        for conv in self.gnn:
            if isinstance(conv, EdgeSAGEConv):
                h = self.act(conv(h, edge_index, edge_attr))
            else:
                h = self.act(conv(h, edge_index))
        
        return h, batch