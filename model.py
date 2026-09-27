"""
PRAVAAH-NOW — Multi-Variable U-Net with a ConvLSTM bottleneck.

Architecture summary (matches Slide 3 / Slide 2 of our PPT):
  - Encoder: standard U-Net downsampling path over the fused
    spatio-temporal grid (radar reflectivity, satellite channels,
    DEM-derived slope/aspect).
  - Bottleneck: a ConvLSTM cell tracks the TEMPORAL evolution of the
    encoded storm representation across the input sequence (this is
    what lets us predict storm GROWTH, not just advect existing
    reflectivity forward like classic extrapolation nowcasting).
  - Decoder: standard U-Net upsampling path back to full-resolution
    rainfall-accumulation / hazard-probability output.

This is a from-scratch, framework-idiomatic PyTorch implementation.
Channel counts and depth are intentionally modest so this trains on a
single consumer/cloud GPU within a hackathon timeline — see
docs/references.md for the papers this design draws from.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv layers + BatchNorm + ReLU, the standard U-Net unit."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell (Shi et al., 2015 formulation).

    Operates on the encoded feature maps at the U-Net bottleneck,
    convolving over space while maintaining LSTM-style gating over
    time — this is what captures thermodynamic storm-growth signals
    that a purely spatial (non-recurrent) bottleneck would miss.
    """

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        # Single conv produces all 4 gates (input, forget, output, cell) at once
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        h, w = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_ch, h, w, device=device),
            torch.zeros(batch_size, self.hidden_ch, h, w, device=device),
        )


class PravaahNowNet(nn.Module):
    """
    Full model: U-Net encoder -> ConvLSTM bottleneck (over T timesteps)
    -> U-Net decoder -> hazard/rainfall output map.

    Args:
        in_channels: number of fused input channels per timestep
            (e.g. radar dBZ, radial velocity, satellite IR/WV channels,
            DEM slope, DEM aspect).
        base_ch: base channel width for the U-Net (doubles each downsample).
        lstm_hidden: hidden channel width of the ConvLSTM bottleneck.
        out_channels: output channels — e.g. 1 for rainfall accumulation (mm),
            or >1 if jointly predicting a hail-probability channel.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_ch: int = 32,
        lstm_hidden: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # ---- ConvLSTM bottleneck ----
        self.lstm_cell = ConvLSTMCell(base_ch * 4, lstm_hidden)

        # ---- Decoder ----
        self.up2 = nn.ConvTranspose2d(lstm_hidden, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)  # + skip connection
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)  # + skip connection

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, C, H, W) — a sequence of T fused input frames.

        Returns:
            (B, out_channels, H, W) — predicted hazard/rainfall map for
            the target lead time.
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        h, c = None, None
        skip1, skip2 = None, None  # kept from the LAST timestep's encoder pass

        for t in range(T):
            xt = x_seq[:, t]  # (B, C, H, W)

            e1 = self.enc1(xt)
            p1 = self.pool(e1)
            e2 = self.enc2(p1)
            p2 = self.pool(e2)
            e3 = self.enc3(p2)  # bottleneck-resolution features

            if h is None:
                h, c = self.lstm_cell.init_hidden(B, e3.shape[-2:], device)
            h, c = self.lstm_cell(e3, h, c)

            skip1, skip2 = e1, e2  # retain most recent skip connections

        # ---- Decode from the FINAL ConvLSTM hidden state ----
        d2 = self.up2(h)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out


if __name__ == "__main__":
    # Quick shape sanity check — run with: python model/model.py
    model = PravaahNowNet(in_channels=6, out_channels=1)
    dummy = torch.randn(2, 4, 6, 128, 128)  # batch=2, T=4 frames, 6 channels, 128x128
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 1, 128, 128)
