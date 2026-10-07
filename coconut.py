# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values"], defaults=[None]
)
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        self.embedding = self.base_causallm.get_input_embeddings()

    @property
    def is_ouro(self):
        return getattr(self.base_causallm.config, "model_type", None) == "ouro"

    def _forward_chunk(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        compute_logits=True,
        use_cache=False,
    ):
        """Return final-step logits and hidden states without using a KV cache.

        Decoder-style Hugging Face models expose a base model's final hidden
        state directly. Ouro instead returns its recurrent-step hidden states
        as an additional tuple, so it needs a small adapter. Keeping this path
        cache-free is slower, but it is robust across cache implementations and
        retains the full autograd graph required by Coconut training.
        """
        if self.is_ouro:
            decoder_outputs, recurrent_hidden_states, _ = self.base_causallm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )
            if not recurrent_hidden_states:
                raise RuntimeError("Ouro did not return recurrent hidden states")
            hidden_states = recurrent_hidden_states[-1]
            logits = (
                self.base_causallm.get_output_embeddings()(hidden_states)
                if compute_logits
                else None
            )
            return logits, hidden_states, decoder_outputs.past_key_values

        decoder = getattr(self.base_causallm, "model", None)
        if decoder is not None:
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            logits = (
                self.base_causallm.get_output_embeddings()(hidden_states)
                if compute_logits
                else None
            )
            return logits, hidden_states, outputs.past_key_values

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        if outputs.hidden_states is None:
            raise RuntimeError(
                f"{type(self.base_causallm).__name__} did not return hidden states"
            )
        return outputs.logits, outputs.hidden_states[-1], outputs.past_key_values

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        output_full_logits=True,
        use_cache=False,
        **kwargs,
    ):

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        inputs_embeds = self.embedding(input_ids)

        for pass_idx in range(max_n_latents):
            active_latent_positions = [
                latent_list[pass_idx]
                for latent_list in latent_lists
                if len(latent_list) > pass_idx
            ]
            prefix_end = max(active_latent_positions)
            _, hidden_states, _ = self._forward_chunk(
                inputs_embeds=inputs_embeds[:, :prefix_end, :],
                attention_mask=attention_mask[:, :prefix_end],
                position_ids=position_ids[:, :prefix_end],
                compute_logits=False,
            )

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            batch_indices, token_indices = zip(*filling_indices)
            batch_indices = torch.tensor(batch_indices, device=inputs_embeds.device)
            token_indices = torch.tensor(token_indices, device=inputs_embeds.device)
            replacements = hidden_states[batch_indices, token_indices - 1]

            # Clone before indexed assignment so autograd preserves both the
            # untouched token embeddings and the continuous-thought graph.
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[batch_indices, token_indices] = replacements

        # One full pass produces the supervised logits after all latent inputs
        # have been filled. Labels for the question and latent positions are
        # masked, so no loss-bearing logits are discarded.
        logits, final_hidden_states, past_key_values = self._forward_chunk(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compute_logits=output_full_logits,
            use_cache=use_cache,
        )

        self.gen_forward_cnt += max_n_latents + 1

        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        if output_full_logits:
            shift_logits = logits[..., :-1, :].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        else:
            supervised = shift_labels != -100
            if not supervised.any():
                raise ValueError("Coconut batch contains no supervised tokens")
            supervised_hidden = final_hidden_states[..., :-1, :][supervised]
            supervised_logits = self.base_causallm.get_output_embeddings()(
                supervised_hidden
            )
            loss = loss_fct(supervised_logits, shift_labels[supervised])

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            past_key_values=past_key_values,
        )

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        use_generation_cache=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
            use_cache=use_generation_cache,
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        past_key_values = outputs.past_key_values
        for _ in range(max_new_tokens - 1):
            if use_generation_cache:
                model_kwargs = {
                    "inputs_embeds": new_token_embed,
                    "attention_mask": torch.ones(
                        (1, new_inputs_embeds.shape[1]),
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    "position_ids": torch.tensor(
                        [[new_inputs_embeds.shape[1] - 1]],
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "logits_to_keep": 1,
                }
                if self.is_ouro:
                    model_kwargs["exit_at_step"] = (
                        self.base_causallm.config.total_ut_steps - 1
                    )
                outputs = self.base_causallm(**model_kwargs)
                past_key_values = outputs.past_key_values
            else:
                outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
