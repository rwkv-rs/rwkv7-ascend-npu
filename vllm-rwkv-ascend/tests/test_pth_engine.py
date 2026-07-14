import pytest
import torch
import torch.nn.functional as F

from perf.rwkv7_pth_engine import (
    make_folded_mix_project_weight,
    pack_lowrank_layer,
)


def test_unequal_lowrank_pack_matches_four_independent_chains():
    torch.manual_seed(20260714)
    hidden = 8
    ranks = (2, 3, 4, 1)
    first = [torch.randn(rank, hidden) for rank in ranks]
    second = [torch.randn(hidden, rank) for rank in ranks]
    biases = [torch.randn(hidden) for _ in ranks]
    inputs = torch.randn(4, 2, hidden)

    first_bmm, second_bmm, bias_bmm, target_rank = pack_lowrank_layer(
        first, second, biases
    )
    intermediate = torch.bmm(inputs, first_bmm)
    intermediate[0] = torch.tanh(intermediate[0])
    intermediate[2] = torch.sigmoid(intermediate[2])
    actual = torch.bmm(intermediate, second_bmm) + bias_bmm

    activations = (torch.tanh, lambda value: value, torch.sigmoid, lambda value: value)
    expected = torch.stack(
        [
            F.linear(activation(F.linear(inputs[index], first[index])), second[index], biases[index])
            for index, activation in enumerate(activations)
        ]
    )

    assert target_rank == max(ranks)
    assert first_bmm.shape == (4, hidden, target_rank)
    assert second_bmm.shape == (4, target_rank, hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_folded_mix_project_matches_independent_shift_mix_projections():
    torch.manual_seed(20260714)
    hidden = 8
    output_sizes = (8, 8, 8, 2, 3, 4, 1)
    weights = [torch.randn(output, hidden) for output in output_sizes]
    mixes = [torch.rand(hidden) for _ in output_sizes]
    current = torch.randn(2, hidden)
    previous = torch.randn(2, hidden)

    folded = make_folded_mix_project_weight(weights, mixes)
    actual = F.linear(torch.cat((current, previous), dim=1), folded)

    target_rank = max(output_sizes[3:])
    expected_groups = []
    for index, (weight, mix) in enumerate(zip(weights, mixes)):
        group = F.linear(current + (previous - current) * mix, weight)
        if index >= 3:
            group = F.pad(group, (0, target_rank - group.shape[1]))
        expected_groups.append(group)
    expected = torch.cat(expected_groups, dim=1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_lowrank_pack_rejects_incompatible_shapes():
    hidden = 8
    first = [torch.randn(2, hidden) for _ in range(4)]
    second = [torch.randn(hidden, 2) for _ in range(4)]
    second[2] = torch.randn(hidden + 1, 2)
    biases = [torch.randn(hidden) for _ in range(4)]

    with pytest.raises(ValueError, match="incompatible shapes"):
        pack_lowrank_layer(first, second, biases)
