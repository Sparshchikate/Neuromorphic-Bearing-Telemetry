import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem_minus_th, gamma=2.0):
        ctx.save_for_backward(mem_minus_th)
        ctx.gamma = gamma
        return (mem_minus_th >= 0.0).to(mem_minus_th.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (mem_minus_th,) = ctx.saved_tensors
        grad = 1.0 / ((1.0 + ctx.gamma * torch.abs(mem_minus_th)) ** 2)
        return grad_output * grad, None


def spike_fn(x, gamma=2.0):
    return FastSigmoidSpike.apply(x, gamma)


class RecurrentLIFLayer(nn.Module):
    def __init__(self, in_features, hidden_features, beta=0.85, u_th=1.00, gamma=2.0, recurrent=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.beta = beta
        self.u_th = u_th
        self.gamma = gamma
        self.recurrent = recurrent

        self.ff = nn.Linear(in_features, hidden_features, bias=False)
        if recurrent:
            self.rec = nn.Linear(hidden_features, hidden_features, bias=False)
            nn.init.orthogonal_(self.rec.weight, gain=0.2)
        else:
            self.rec = None

        nn.init.kaiming_normal_(self.ff.weight, nonlinearity="linear")

    def forward(self, x_seq):
        # x_seq: (batch, time_steps, in_features)
        batch_size, time_steps, _ = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        u = torch.zeros(batch_size, self.hidden_features, device=device, dtype=dtype)
        s_prev = torch.zeros(batch_size, self.hidden_features, device=device, dtype=dtype)
        ff_in = self.ff(x_seq)

        spikes = []
        mem_trace = []

        for t in range(time_steps):
            syn_in = ff_in[:, t, :]
            if self.recurrent:
                syn_in = syn_in + self.rec(s_prev)

            # Eq. (9): subtractive reset from previous step's spike
            u = self.beta * u + syn_in - s_prev * self.u_th
            s = spike_fn(u - self.u_th, self.gamma)

            spikes.append(s)
            mem_trace.append(u)
            s_prev = s

        return torch.stack(spikes, dim=1), torch.stack(mem_trace, dim=1)


class NeuromorphicRLIFClassifier(nn.Module):
    def __init__(self, in_channels=2, h1=128, h2=64, num_classes=10, beta=0.85, u_th=1.00, gamma=2.0, recurrent=True):
        super().__init__()
        self.layer1 = RecurrentLIFLayer(in_channels, h1, beta=beta, u_th=u_th, gamma=gamma, recurrent=recurrent)
        self.layer2 = RecurrentLIFLayer(h1, h2, beta=beta, u_th=u_th, gamma=gamma, recurrent=recurrent)
        self.readout = nn.Linear(h2, num_classes, bias=True)

    def forward(self, x_seq, return_stats=False):
        s1, u1 = self.layer1(x_seq)
        s2, u2 = self.layer2(s1)

        # Eq. (10): cumulative spike-count logits over window W
        spike_counts = s2.sum(dim=1)
        logits = self.readout(spike_counts)

        if not return_stats:
            return logits

        stats = {
            "in_spikes": x_seq.detach(),
            "s1": s1.detach(),
            "s2": s2.detach(),
            "u1": u1.detach(),
            "u2": u2.detach(),
        }
        return logits, stats


class Baseline1DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq):
        # x_seq: (batch, time_steps, 1) -> (batch, 1, time_steps)
        x = x_seq.transpose(1, 2)
        return self.classifier(self.features(x))