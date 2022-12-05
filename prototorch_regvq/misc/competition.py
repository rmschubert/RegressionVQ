import torch


class WTAC_regression(torch.nn.Module):
    def __init__(self):
        super(WTAC_regression, self).__init__()

    def wtac_regression(
        self,
        reg_vals: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:

        _, winning_indices = torch.min(distances, 1)

        preds = torch.squeeze(torch.gather(reg_vals, 1, winning_indices.view(-1, 1)))

        return preds

    def forward(self, reg_vals, distances):
        return self.wtac_regression(reg_vals, distances)


class WTAC_RLVQ(torch.nn.Module):
    def __init__(self):
        super(WTAC_RLVQ, self).__init__()
    
    def wtac_rlvq(
        self,
        probabilities: torch.Tensor,
        approximations: torch.Tensor,
        soft: bool,
    ) -> torch.Tensor:

        
        if soft:
            return torch.sum(approximations * probabilities, 1)
        else:
            _, winning_indices = torch.max(probabilities, 1)
            winner_preds = approximations[winning_indices]
            return winner_preds
    
    def forward(self, probabilities, approximations, soft):
        return self.wtac_rlvq(probabilities, approximations, soft)


