import torch
import pandas as pd
from typing import List


def GCD(lat1: torch.tensor, lng1: torch.tensor, lat2: torch.tensor, lng2: torch.tensor) -> torch.tensor:
    """
    Calculates the Great-Circle Distance (in km).
    """
    lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])

    dlng = lng2 - lng1
    dlat = lat2 - lat1

    a = torch.sin(dlat/2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng/2.0)**2
    c = 2 * torch.asin(torch.sqrt(a))

    return 6371 * c


def acc(GCD: torch.tensor, thresholds: List[int] = [1, 25, 200, 750, 2500]) -> dict:
    """
    Calculates accuracies for various GCD thresolds (in km).
    """
    results = {}
    
    for threshold in thresholds:
        results[threshold] = torch.sum(GCD <= threshold).item() / len(GCD) * 100

    return results
