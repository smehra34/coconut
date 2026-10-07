"""Numerical equivalence checks for Coconut's training hot path."""

import copy
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coconut import Coconut


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


class ToyDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds, **kwargs):
        # Causal mixing makes later latent replacements depend on earlier ones.
        hidden = torch.tanh(self.proj(inputs_embeds.cumsum(dim=1)))
        return SimpleNamespace(last_hidden_state=hidden)


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size=13, hidden_size=7):
        super().__init__()
        self.config = SimpleNamespace(model_type="toy")
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = ToyDecoder(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_projection_calls = 0

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        parent = self

        class CountedProjection:
            def __call__(self, hidden_states):
                parent.output_projection_calls += 1
                return parent.lm_head(hidden_states)

        return CountedProjection()


class ReferenceCoconut(Coconut):
    """The pre-optimization forward path used as a numerical oracle."""

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max(len(indices) for indices in latent_lists)
        inputs_embeds = self.embedding(input_ids)

        for pass_idx in range(max_n_latents):
            active_positions = [
                indices[pass_idx]
                for indices in latent_lists
                if len(indices) > pass_idx
            ]
            prefix_end = max(active_positions)
            _, hidden_states = self._forward_chunk(
                inputs_embeds[:, :prefix_end],
                attention_mask[:, :prefix_end],
                position_ids[:, :prefix_end],
            )
            tensor_list = [
                [inputs_embeds[b, pos] for pos in range(inputs_embeds.shape[1])]
                for b in range(inputs_embeds.shape[0])
            ]
            for batch_idx, indices in enumerate(latent_lists):
                if len(indices) > pass_idx:
                    token_idx = indices[pass_idx]
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, token_idx - 1
                    ]
            inputs_embeds = torch.stack(
                [torch.stack(instance) for instance in tensor_list]
            )

        logits, _ = self._forward_chunk(
            inputs_embeds, attention_mask, position_ids
        )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return Outputs(loss, inputs_embeds, logits)


class CoconutOptimizationTest(unittest.TestCase):
    def test_forward_and_backward_match_reference(self):
        torch.manual_seed(7)
        optimized_base = ToyCausalLM()
        reference_base = copy.deepcopy(optimized_base)
        optimized = Coconut(optimized_base, 10, 11, 12, 1)
        reference = ReferenceCoconut(reference_base, 10, 11, 12, 1)

        # Instance 0 has two latent positions; instance 1 has one at a
        # different position, exercising the uneven-batch update path.
        input_ids = torch.tensor([[2, 3, 10, 10, 4, 5, 1], [6, 7, 8, 10, 9, 4, 1]])
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        labels = input_ids.clone()
        labels[:, :4] = -100

        actual = optimized(input_ids, attention_mask, labels, position_ids)
        expected = reference(input_ids, attention_mask, labels, position_ids)

        torch.testing.assert_close(actual.inputs_embeds, expected.inputs_embeds)
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.loss, expected.loss)
        self.assertEqual(optimized_base.output_projection_calls, 1)
        self.assertEqual(reference_base.output_projection_calls, 3)

        actual.loss.backward()
        expected.loss.backward()
        for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
            optimized.named_parameters(), reference.named_parameters()
        ):
            self.assertEqual(actual_name, expected_name)
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                msg=lambda message: f"gradient mismatch for {actual_name}: {message}",
            )


if __name__ == "__main__":
    unittest.main()
