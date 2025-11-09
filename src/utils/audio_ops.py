import math
import torch


def overlap_and_add(signal: torch.Tensor, frame_step: int) -> torch.Tensor:
    """
    Reconstruct a 1D signal from overlapping frames.

    Args:
        signal: Tensor with shape [..., frames, frame_length]
        frame_step: Step between frames (hop). Must be <= frame_length.

    Returns:
        Tensor with shape [..., output_size], where
        output_size = (frames - 1) * frame_step + frame_length
    """
    outer_dims = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # reshape last two dims into subframes
    subframe_signal = signal.reshape(*outer_dims, -1, subframe_length)

    # indices for overlap-add
    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().contiguous().view(-1)

    result = signal.new_zeros(*outer_dims, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dims, -1)
    return result
