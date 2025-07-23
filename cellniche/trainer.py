import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import tqdm

from model import Model, Encoder
from sampler import NeighborSampler
from utils import setup_seed, load_data, get_positivePairs, clustering_st

def run(args):
    """
    Main training and inference pipeline.
    """
    load_start = time.time()
    # Set random seed for reproducibility
    setup_seed(args.seed)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data and graph structure
    x, edge_index, y, adata, n_classes, expr = load_data(
        data_path=args.data_path,
        dataset=args.dataset,
        phenoLabels=args.phenoLabels,
        nicheLabels=args.nicheLabels,
        embedding_type=args.embedding_type,
        radius=args.radius,
        k_neighborhood=args.k_neighborhood,
        hvg=args.hvg,
    )
    x = x.to(device)
    if expr is not None:
        expr = expr.to(device)
        
    # Build sparse adjacency tensor
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones(edge_index.shape[1]),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )

    # Create NeighborSampler loaders
    train_loader = NeighborSampler(
        edge_index,
        adj,
        is_train=True,
        node_idx=None,
        wt=args.wt,
        wl=args.wl,
        p=args.p,
        q=args.q,
        sizes=args.size,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    test_loader = NeighborSampler(
        edge_index,
        adj,
        is_train=False,
        node_idx=None,
        sizes=args.size,
        batch_size=args.batch_size*4,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    # Initialize encoder and model
    encoder = Encoder(
        in_channels=x.shape[1],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        negative_slope=args.negative_slope,
    ).to(device)
    model = Model(
        encoder=encoder,
        decoder_hidden=args.decoder,
        project_hidden=args.projection,
        use_weight=args.use_weight,
        pos_weight_strategy=args.pos_weight_strategy,
        neg_weight_strategy=args.neg_weight_strategy,
    ).to(device)
    logging.info(f"Model architecture:\n{model}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # load_end = time.time()
    # logging.info(
    #     f"loading_time: {load_end-load_start:.2f}s"
    # )
    
    # Training loop
    train_start = time.time()
    step = 0
    
    use_epochs = args.epochs is not None and args.epochs > 0
    use_steps  = not use_epochs and args.max_steps is not None and args.max_steps > 0
    
    if use_epochs:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for (bs, n_id, adjs), adj_batch, *rest in train_loader:
                # Move subgraph to device
                adjs = [adj.to(device) for adj in adjs]
                adj_batch = adj_batch.to(device)

                # Generate positive mask
                features = expr if args.embedding_type == "pheno_expr" else x
                pos_mask = get_positivePairs(adj_batch, features, strategy=args.strategy)

                optimizer.zero_grad()
                z, proj_z, rec_expr, _ = model(x[n_id], adjs=adjs)
                z = F.normalize(z, p=2, dim=1)
                proj_z = F.normalize(proj_z, p=2, dim=1)
                contrast_loss, recon_loss = model.compute_loss(
                    z, pos_mask, proj_z, rec_expr, target=expr
                )
                loss = 0.8 * contrast_loss + 0.2 * recon_loss if args.decoder else contrast_loss
                loss.backward()
                optimizer.step()

                step += 1
                if args.verbose:
                    logging.info(
                        f"Epoch {epoch} Step {step:04d} "
                        f"contrast_loss={contrast_loss:.4f}, recon_loss={recon_loss:.4f}"
                    )

    elif use_steps:
        while step < args.max_steps:
            model.train()
            for (bs, n_id, adjs), adj_batch, *rest in train_loader:
                # Move subgraph to device
                adjs = [adj.to(device) for adj in adjs]
                adj_batch = adj_batch.to(device)

                # Generate positive mask
                features = expr if args.embedding_type == "pheno_expr" else x
                pos_mask = get_positivePairs(adj_batch, features, strategy=args.strategy)

                optimizer.zero_grad()
                z, proj_z, rec_expr, _ = model(x[n_id], adjs=adjs)
                z = F.normalize(z, p=2, dim=1)
                proj_z = F.normalize(proj_z, p=2, dim=1)
                contrast_loss, recon_loss = model.compute_loss(
                    z, pos_mask, proj_z, rec_expr, target=expr
                )
                loss = 0.8 * contrast_loss + 0.2 * recon_loss if args.decoder else contrast_loss
                loss.backward()
                optimizer.step()

                step += 1
                if args.verbose:
                    logging.info(
                        f"Step {step:04d} "
                        f"contrast_loss={contrast_loss:.4f}, recon_loss={recon_loss:.4f}"
                    )
    else:
        logging.warning("Neither epochs nor max_steps specified, no training performed.")
    
    # logging.info(f"Training completed in {time.time() - train_start:.2f}s")

    # Inference: compute embeddings for all nodes
    model.eval()
    z_all = torch.zeros((x.shape[0], args.hidden_channels[-1]), device='cpu')
    with torch.no_grad():
        for (bs, n_id, adjs), _, batch in tqdm(test_loader, desc="Inference"):    
            
            adjs = [adj.to(device) for adj in adjs]
            z, _, _, _ = model(x[n_id].to(device), adjs=adjs)
            
            z_all[batch] = z.detach().cpu().float()

    z_all = F.normalize(z_all, p=2, dim=1)
    adata.obsm["CellNiche"] = z_all.numpy()

    # Clustering and metrics
    if args.metrics:
        adata, metrics = clustering_st(adata, n_classes, z_all, y, refine=args.refine)
        logging.info(f"Clustering metrics: {metrics}")
    # Save results
    if args.save:
        output_file = f"{args.save_path}/{args.dataset}_emb_{args.embedding_type}.h5ad"
        adata.write_h5ad(output_file)
        logging.info(f"Saved embeddings to {output_file}")
