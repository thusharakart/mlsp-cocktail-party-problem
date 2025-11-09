import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils.audio_ops import overlap_and_add


class Encoder(nn.Module):
    """Waveform encoder using 1D conv with 50% overlap.
    Input: [B, T] -> Output: [B, N, L]
    """

    def __init__(self, W: int = 2, N: int = 64):
        super().__init__()
        self.W, self.N = W, N
        self.conv1d = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        x = mixture.unsqueeze(1)  # [B, 1, T]
        return F.relu(self.conv1d(x))  # [B, N, L]


class Decoder(nn.Module):
    """Linear synthesis from encoded basis to waveform via overlap-and-add.
    mixture_w: [B, E, L], est_mask: [B, C, E, L] -> est_source: [B, C, T]
    """

    def __init__(self, E: int, W: int):
        super().__init__()
        self.E, self.W = E, W
        self.basis = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w: torch.Tensor, est_mask: torch.Tensor) -> torch.Tensor:
        # [B, C, E, L]
        source_w = mixture_w.unsqueeze(1) * est_mask
        # [B, C, L, E]
        source_w = source_w.transpose(2, 3)
        # [B, C, L, W]
        est_frames = self.basis(source_w)
        # overlap-add with hop W//2 -> [B, C, T]
        return overlap_and_add(est_frames, self.W // 2)


class SingleRNN(nn.Module):
    """One RNN layer + linear projection operating on sequences."""

    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        num_directions = 2 if bidirectional else 1
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * num_directions, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        B, T, D = out.shape
        out = self.proj(out.contiguous().view(B * T, D)).view(B, T, -1)
        return out


class DPRNN(nn.Module):
    """Dual-Path RNN block applied along intra- and inter-chunk dimensions.

    Input shape: [B, N, L, K] -> Output: [B, output_size, L, K]
    """

    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0, num_layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.row_rnn = nn.ModuleList()
        self.col_rnn = nn.ModuleList()
        self.row_norm = nn.ModuleList()
        self.col_norm = nn.ModuleList()
        for _ in range(num_layers):
            # intra-chunk (row) RNN is bidirectional (noncausal)
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True))
            # inter-chunk (col) RNN may be bidirectional
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(input_size, output_size, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L, K]
        B, N, L, K = x.shape
        out = x
        for row_rnn, col_rnn, row_norm, col_norm in zip(self.row_rnn, self.col_rnn, self.row_norm, self.col_norm):
            # Intra-chunk along L (row)
            row_in = out.permute(0, 3, 2, 1).contiguous().view(B * K, L, N)
            row_out = row_rnn(row_in)
            row_out = row_out.view(B, K, L, N).permute(0, 3, 2, 1).contiguous()  # [B, N, L, K]
            out = out + row_norm(row_out)

            # Inter-chunk along K (col)
            col_in = out.permute(0, 2, 3, 1).contiguous().view(B * L, K, N)
            col_out = col_rnn(col_in)
            col_out = col_out.view(B, L, K, N).permute(0, 3, 1, 2).contiguous()  # [B, N, L, K]
            out = out + col_norm(col_out)

        return self.output(out)  # [B, output_size, L, K]


class BFModule(nn.Module):
    """Beamforming-like module on encoded features using DPRNN blocks."""

    def __init__(self, input_dim: int, feature_dim: int, hidden_dim: int, num_spk: int = 2, layers: int = 4, segment_size: int = 100, bidirectional: bool = True, rnn_type: str = 'LSTM'):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_spk = num_spk
        self.layers = layers
        self.segment_size = segment_size

        # bottleneck 1x1 conv: [B, E, L] -> [B, N, L]
        self.bottleneck = nn.Conv1d(self.input_dim, self.feature_dim, kernel_size=1, bias=False)

        self.dprnn = DPRNN(rnn_type, self.feature_dim, self.hidden_dim, self.feature_dim * self.num_spk, num_layers=layers, bidirectional=bidirectional)

        # gated output layers to produce filters
        self.out = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Tanh())
        self.out_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Sigmoid())

    def pad_segment(self, x: torch.Tensor, segment_size: int):
        B, N, T = x.shape
        stride = segment_size // 2
        rest = segment_size - (stride + T % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(B, N, rest, dtype=x.dtype, device=x.device))
            x = torch.cat([x, pad], dim=2)
        pad_aux = Variable(torch.zeros(B, N, stride, dtype=x.dtype, device=x.device))
        x = torch.cat([pad_aux, x, pad_aux], dim=2)
        return x, rest

    def split_feature(self, x: torch.Tensor, segment_size: int):
        x, rest = self.pad_segment(x, segment_size)
        B, N, T = x.shape
        stride = segment_size // 2
        seg1 = x[:, :, :-stride].contiguous().view(B, N, -1, segment_size)
        seg2 = x[:, :, stride:].contiguous().view(B, N, -1, segment_size)
        segs = torch.cat([seg1, seg2], dim=3).view(B, N, -1, segment_size).transpose(2, 3)
        return segs.contiguous(), rest

    def merge_feature(self, x: torch.Tensor, rest: int):
        # x: [B, N, L, K]
        B, N, L, K = x.shape
        stride = L // 2
        x = x.transpose(2, 3).contiguous().view(B, N, -1, L * 2)  # [B, N, K, 2L]
        x1 = x[:, :, :, :L].contiguous().view(B, N, -1)[:, :, stride:]
        x2 = x[:, :, :, L:].contiguous().view(B, N, -1)[:, :, :-stride]
        out = x1 + x2
        if rest > 0:
            out = out[:, :, :-rest]
        return out.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E, T]
        B, E, T = x.shape
        enc_feat = self.bottleneck(x)  # [B, N, T]
        segs, rest = self.split_feature(enc_feat, self.segment_size)  # [B, N, L, K]
        out = self.dprnn(segs).view(B * self.num_spk, self.feature_dim, self.segment_size, -1)
        out = self.merge_feature(out, rest)  # [B*num_spk, N, T]
        filt = self.out(out) * self.out_gate(out)  # gated filters, [B*num_spk, N, T]
        return filt.transpose(1, 2).contiguous().view(B, self.num_spk, -1, self.feature_dim)  # [B, C, T, N]


class DPRNNSeparator(nn.Module):
    """End-to-end DPRNN-based time-domain separator (FaSNet-style).

    Forward signature matches existing pipeline: mixture [B, T] -> separated [B, C, T]
    """

    def __init__(self, num_sources: int = 2, enc_dim: int = 256, feature_dim: int = 64, hidden_dim: int = 128, layers: int = 6, segment_size: int = 250, win_len: int = 2, rnn_type: str = 'LSTM'):
        super().__init__()
        self.num_sources = num_sources
        self.enc = Encoder(W=win_len, N=enc_dim)
        self.enc_norm = nn.GroupNorm(1, enc_dim, eps=1e-8)
        self.separator = BFModule(input_dim=enc_dim, feature_dim=feature_dim, hidden_dim=hidden_dim, num_spk=num_sources, layers=layers, segment_size=segment_size, rnn_type=rnn_type)
        self.mask_1x1 = nn.Conv1d(feature_dim, enc_dim, kernel_size=1, bias=False)
        self.dec = Decoder(enc_dim, win_len)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # mixture: [B, T]
        B = mixture.size(0)
        mixture_w = self.enc(mixture)  # [B, E, L]
        score = self.enc_norm(mixture_w)
        bf_filters = self.separator(score)  # [B, C, T, N]
        score = bf_filters.view(B * self.num_sources, -1, self.separator.feature_dim).transpose(1, 2).contiguous()  # [B*C, N, T]
        score = self.mask_1x1(score)  # [B*C, E, L]
        score = score.view(B, self.num_sources, -1, score.size(-1))  # [B, C, E, L]
        est_mask = F.relu(score)
        est_source = self.dec(mixture_w, est_mask)  # [B, C, T]
        return est_source
