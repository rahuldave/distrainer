import numpy as np
import torch

from distrainer.config import DistrainerConfig
from examples.toy_contrastive.make_blocks import hard_negative_clusters, make_corpus, mine_rows
from examples.toy_contrastive.model import Encoder, info_nce


def small_cfg():
    return DistrainerConfig.from_dict(
        {"seed": 3, "train": {"n_items": 96, "features": 6, "clusters": 8, "hard_negatives": 3}}
    )


def test_corpus_and_hard_negatives_are_deterministic_and_well_formed():
    items, cluster, centroids = make_corpus(small_cfg())
    items2, cluster2, _ = make_corpus(small_cfg())
    assert items.shape == (96, 6) and centroids.shape == (8, 6) and cluster.shape == (96,)
    assert np.array_equal(items, items2) and np.array_equal(cluster, cluster2)
    near = hard_negative_clusters(centroids, 3)
    assert near.shape == (8, 3)
    for c in range(8):
        assert c not in near[c], "a cluster is never its own hard negative"
        assert len(set(near[c])) == 3


def test_mine_rows_positive_same_cluster_negatives_from_near_clusters():
    items, cluster, centroids = make_corpus(small_cfg())
    near = hard_negative_clusters(centroids, 3)
    mined = mine_rows(items, cluster, near, 3, np.random.default_rng(0))
    pos, negs = mined["positive"], mined["negatives"]
    assert pos.shape == (96,) and negs.shape == (96, 3)
    for i in range(96):
        assert cluster[pos[i]] == cluster[i]
        for j in range(3):
            assert cluster[negs[i, j]] == near[cluster[i]][j]


def test_info_nce_prefers_the_positive():
    torch.manual_seed(0)
    enc = Encoder(6, 16, 8)
    anchor = torch.randn(5, 6)
    z = enc(anchor)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    positive = z + 0.01 * torch.randn_like(z)
    negatives = torch.randn(5, 3, 8)
    good = info_nce(z, positive, negatives, temperature=0.1)
    bad = info_nce(z, torch.randn_like(z), negatives, temperature=0.1)
    assert good.item() < bad.item()
    good.backward()  # gradients flow to the encoder
    assert all(p.grad is not None for p in enc.parameters())
