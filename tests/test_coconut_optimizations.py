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

    def forward(
        self, inputs_embeds, past_key_values=None, use_cache=False, **kwargs
    ):
        # Causal mixing makes later latent replacements depend on earlier ones.
        if past_key_values is not None:
            all_inputs = torch.cat((past_key_values, inputs_embeds), dim=1)
        else:
            all_inputs = inputs_embeds
        hidden = torch.tanh(self.proj(all_inputs.cumsum(dim=1)))
        if past_key_values is not None:
            hidden = hidden[:, -inputs_embeds.shape[1] :]
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=all_inputs if use_cache else None,
        )


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

    def forward(
        self,
        inputs_embeds,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        outputs = self.model(
            inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return SimpleNamespace(
            logits=self.lm_head(outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
        )


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
            _, hidden_states, _ = self._forward_chunk(
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

        logits, _, _ = self._forward_chunk(
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

    def test_selective_logits_match_full_loss_and_gradients(self):
        torch.manual_seed(11)
        full = Coconut(ToyCausalLM(), 10, 11, 12, 1)
        selective = copy.deepcopy(full)
        input_ids = torch.tensor(
            [[2, 3, 10, 10, 4, 5, 1], [6, 7, 8, 10, 9, 4, 1]]
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        labels = input_ids.clone()
        labels[:, :4] = -100

        full_output = full(input_ids, attention_mask, labels, position_ids)
        selective_output = selective(
            input_ids,
            attention_mask,
            labels,
            position_ids,
            output_full_logits=False,
        )
        self.assertIsNone(selective_output.logits)
        torch.testing.assert_close(selective_output.loss, full_output.loss)

        full_output.loss.backward()
        selective_output.loss.backward()
        for (full_name, full_parameter), (
            selective_name,
            selective_parameter,
        ) in zip(full.named_parameters(), selective.named_parameters()):
            self.assertEqual(full_name, selective_name)
            torch.testing.assert_close(
                selective_parameter.grad,
                full_parameter.grad,
                msg=lambda message: f"gradient mismatch for {full_name}: {message}",
            )

    def test_generation_cache_matches_full_prefix_generation(self):
        torch.manual_seed(19)
        model = Coconut(ToyCausalLM(), 10, 11, 12, 1).eval()
        input_ids = torch.tensor([[2, 3, 10, 10, 4]])
        attention_mask = torch.ones_like(input_ids)

        without_cache = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=8,
            use_generation_cache=False,
        )
        with_cache = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=8,
            use_generation_cache=True,
        )
        torch.testing.assert_close(with_cache, without_cache)

    @unittest.skipUnless(torch.cuda.is_available(), "fused AdamW requires CUDA")
    def test_fused_adamw_matches_unfused_update(self):
        torch.manual_seed(23)
        unfused_parameter = nn.Parameter(torch.randn(31, device="cuda"))
        fused_parameter = nn.Parameter(unfused_parameter.detach().clone())
        unfused = torch.optim.AdamW(
            [unfused_parameter], lr=2e-5, weight_decay=0.01, fused=False
        )
        fused = torch.optim.AdamW(
            [fused_parameter], lr=2e-5, weight_decay=0.01, fused=True
        )

        for _ in range(4):
            gradient = torch.randn_like(unfused_parameter)
            unfused_parameter.grad = gradient.clone()
            fused_parameter.grad = gradient.clone()
            unfused.step()
            fused.step()
            unfused.zero_grad()
            fused.zero_grad()

        torch.testing.assert_close(
            fused_parameter, unfused_parameter, rtol=1e-6, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
