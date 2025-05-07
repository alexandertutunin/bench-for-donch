import torch
torch.manual_seed(1337)

def custom_metric_torch_batch(y_true: torch.Tensor, y_pred: torch.Tensor, beta=1.0, verbose=0) -> torch.Tensor:

    tp = torch.sum((y_pred == 1) & (y_true == 1), dim=-1)
    tn = torch.sum((y_pred == 0) & (y_true == 0), dim=-1)
    fp = torch.sum((y_pred == 1) & (y_true == 0), dim=-1)
    fn = torch.sum((y_pred == 0) & (y_true == 1), dim=-1)

    if verbose:
        print(f'TP: {tp}')
        print(f'TN: {tn}')
        print(f'FP: {fp}')
        print(f'FN: {fn}')
    
    denominator = (1 + beta**2) * tp + fp + (beta**2) *fn
    metric = torch.where(denominator > 0, (1 + beta**2) * tp / denominator, torch.zeros_like(denominator))
    metric = torch.where(tn == y_true.shape[-1], torch.ones_like(tn), metric)
    return metric  # (batch_size,)

def double_metric(y_true: torch.Tensor, y_pred: torch.Tensor, beta=1.0, verbose=0):

    faulty = y_true.sum(dim=-1) > 0
    non_faulty = y_true.sum(dim=-1) == 0

    metric_faulty = custom_metric_torch_batch(
        y_true=y_true[faulty],
        y_pred=y_pred[faulty], 
        beta=beta, 
        verbose=verbose)
    metric_non_faulty = custom_metric_torch_batch(
        y_true=y_true[non_faulty], 
        y_pred=y_pred[non_faulty], 
        beta=beta, 
        verbose=verbose)
    
    return metric_faulty, metric_non_faulty