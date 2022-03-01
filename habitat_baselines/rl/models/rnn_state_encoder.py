from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        r"""An RNN for encoding the state in RL.

        Supports masking the hidden state during various timesteps in the forward lass

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def single_forward(self, x, hidden_states, masks):
        r"""Forward for a non-sequence input
        """
        hidden_states = self._unpack_hidden(hidden_states)
        x, hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        )
        x = x.squeeze(0)
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def seq_forward(self, x, hidden_states, masks):
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1))
        masks = masks.view(t, n)

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]  # handle scalar
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        for i in range(len(has_zeros) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hidden_states = self.rnn(
                x[start_idx:end_idx],
                self._mask_hidden(
                    hidden_states, masks[start_idx].view(1, -1, 1)
                ),
            )

            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)
        x = x.view(t * n, -1)  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)

class BasicRNNStateEncoder(nn.Module):
    r"""RNN encoder for use with RL and possibly IL.

    The main functionality this provides over just using PyTorch's RNN interface directly
    is that it takes an addition masks input that resets the hidden state between two adjacent
    timesteps to handle episodes ending in the middle of a rollout.
    """

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def single_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a non-sequence input"""

        hidden_states = torch.where(
            masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(())
        )

        x, hidden_states = self.rnn(
            x.unsqueeze(0), self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x = x.squeeze(0)
        return x, hidden_states

    def seq_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        N = hidden_states.size(1)

        (
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
        ) = build_rnn_inputs(x, masks, hidden_states)

        x_seq, hidden_states = self.rnn(
            x_seq, self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )

        return x, hidden_states

    def forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) == hidden_states.size(1):
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            x, hidden_states = self.seq_forward(x, hidden_states, masks)

        hidden_states = hidden_states.permute(1, 0, 2)

        return x, hidden_states

def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output

def build_rnn_inputs(
    x: torch.Tensor, not_dones: torch.Tensor, rnn_states: torch.Tensor
) -> Tuple[
    PackedSequence, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param not_dones: A (T * N) tensor where not_dones[i] == False indicates an episode is done
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_episode_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_episode_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """

    N = rnn_states.size(1)
    T = x.size(0) // N
    dones = torch.logical_not(not_dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones.detach().to(device="cpu"), T)

    select_inds = select_inds.to(device=x.device)
    episode_starts = episode_starts.to(device=x.device)
    rnn_state_batch_inds = rnn_state_batch_inds.to(device=x.device)
    last_episode_in_batch_mask = last_episode_in_batch_mask.to(device=x.device)

    x_seq = PackedSequence(
        x.index_select(0, select_inds), batch_sizes, None, None
    )

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, episode_starts),
        rnn_states,
        rnn_states.new_zeros(()),
    )

    return (
        x_seq,
        rnn_states,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states,
    select_inds,
    rnn_state_batch_inds,
    last_episode_in_batch_mask,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_episode_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    last_hidden_states = torch.masked_select(
        hidden_states,
        last_episode_in_batch_mask.view(1, hidden_states.size(1), 1),
    ).view(hidden_states.size(0), N, hidden_states.size(2))
    output_hidden_states = torch.empty_like(last_hidden_states)
    scatter_inds = (
        torch.masked_select(rnn_state_batch_inds, last_episode_in_batch_mask)
        .view(1, N, 1)
        .expand_as(output_hidden_states)
    )
    output_hidden_states.scatter_(1, scatter_inds, last_hidden_states)

    return x, output_hidden_states

class LSTMStateEncoder(BasicRNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers * 2

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def pack_hidden(
        self, hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(hidden_states, 0)

    def unpack_hidden(
        self, hidden_states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_states = torch.chunk(hidden_states, 2, 0)
        return (lstm_states[0], lstm_states[1])

class GRUStateEncoder(BasicRNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()


def build_rnn_state_encoder(
    input_size: int,
    hidden_size: int,
    rnn_type: str = "GRU",
    num_layers: int = 1,
):
    r"""Factory for :ref:`RNNStateEncoder`.  Returns one with either a GRU or LSTM based on
        the specified RNN type.

    :param input_size: The input size of the RNN
    :param hidden_size: The hidden dimension of the RNN
    :param rnn_types: The type of the RNN cell.  Can either be GRU or LSTM
    :param num_layers: The number of RNN layers.
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return GRUStateEncoder(input_size, hidden_size, num_layers)
    elif rnn_type == "lstm":
        return LSTMStateEncoder(input_size, hidden_size, num_layers)
    else:
        raise RuntimeError(f"Did not recognize rnn type '{rnn_type}'")
