import numpy as np
import torch


def hit(ng_item, pred_items):
    return 1 if ng_item in pred_items else 0


def ndcg(ng_item, pred_items):
    if ng_item in pred_items:
        index = pred_items.index(ng_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def calc_metrics(model, test_loader, device, top_k: int = 10):
    HR, NDCG = [], []

    for item_indices in test_loader:
        predictions = model(item_indices)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item_indices, indices).cpu().numpy().tolist()
        ng_item = item_indices[0].item()  # leave one-out evaluation has only one item per user
        HR.append(hit(ng_item, recommends))
        NDCG.append(ndcg(ng_item, recommends))

    return {'HR': np.mean(HR),
            'NDCG': np.mean(NDCG),
            'num_examples': 100
            }
