"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.utils.generation_utils import get_num_transfer_tokens
from dllm.core.generation.generator import GeneratorOutput, GeneratorConfig, BaseGenerator


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def apply_repetition_penalty(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    penalty: float = 1.2,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits in place (returns logits).
    """
    if penalty == 1.0:
        return logits

    device = logits.device
    B = sequences.shape[0]

    if logits.ndim == 3:
        # [B, T_logits, V]
        T_logits = logits.shape[1]
        V = logits.shape[2]
        for b in range(B):
            seq = sequences[b]
            uniq = torch.unique(seq)
            if mask_token_id is not None:
                uniq = uniq[uniq != mask_token_id]
            if eos_token_id is not None:
                uniq = uniq[uniq != eos_token_id]
            if uniq.numel() == 0:
                continue
            uniq = uniq.to(device=device, dtype=torch.long)
            slice_b = logits[b, :, uniq]
            pos_mask = slice_b > 0
            slice_b = torch.where(pos_mask, slice_b / penalty, slice_b * penalty)
            logits[b, :, uniq] = slice_b
    elif logits.ndim == 2:
        # [B, V]
        V = logits.shape[1]
        for b in range(B):
            seq = sequences[b]
            uniq = torch.unique(seq)
            if mask_token_id is not None:
                uniq = uniq[uniq != mask_token_id]
            if eos_token_id is not None:
                uniq = uniq[uniq != eos_token_id]
            if uniq.numel() == 0:
                continue
            uniq = uniq.to(device=device, dtype=torch.long)
            slice_b = logits[b, uniq]
            pos_mask = slice_b > 0
            slice_b = torch.where(pos_mask, slice_b / penalty, slice_b * penalty)
            logits[b, uniq] = slice_b
    else:
        raise ValueError(f"Unexpected logits.ndim={logits.ndim}")

    return logits


@dataclass
class LLaDAGeneratorConfig(GeneratorConfig):
    max_new_tokens: int = 128
    max_length: int = None
    block_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    repetition_penalty: float = 1.2


@dataclass
class LLaDAGenerator(BaseGenerator):
    
    def _apply_stop_criteria(self, x: torch.Tensor):
        """
        Custom Logic:
        Checks for specific stop strings/tokens. If found:
        1. Replaces the stop sequence start with "\n[/Answer]\n".
        2. Replaces all remaining positions after that with <pad>.
        """
        stop_strings = ["|USER|>", "<|USER|>", "<s>", "</s>"]
        target_str = "\n[/Answer]\n"
        
        # Prepare replacement tokens
        replacement_ids = self.tokenizer.encode(target_str, add_special_tokens=False)
        replacement_tensor = torch.tensor(replacement_ids, device=x.device, dtype=torch.long)
        len_repl = len(replacement_ids)
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        B = x.shape[0]
        # Decode with special tokens initially to ensure we capture structure, then rely on string matching
        decoded_texts = self.tokenizer.batch_decode(x, skip_special_tokens=False)

        for b in range(B):
            current_text = decoded_texts[b]
            
            found_stop = None
            min_idx = float('inf')
            
            for s in stop_strings:
                idx = current_text.find(s)
                if idx != -1 and idx < min_idx:
                    min_idx = idx
                    found_stop = s
            
            if found_stop is not None:
                # Calculate token index corresponding to min_idx
                prefix_text = current_text[:min_idx]
                
                # Encode prefix to count tokens
                # Note: add_special_tokens=False is crucial to prevent double BOS counting
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                token_idx = len(prefix_ids)
                
                # --- FIX: BOS Offset Correction ---
                # If the actual tensor x starts with a BOS token (which is standard for many models),
                # but encode(prefix, add_special_tokens=False) does not produce it,
                # our token_idx will be 1 short (pointing to the previous token).
                if x.shape[1] > 0 and self.tokenizer.bos_token_id is not None:
                    if x[b, 0] == self.tokenizer.bos_token_id:
                        # Check if prefix_ids already included BOS (unlikely with add_special_tokens=False)
                        if len(prefix_ids) == 0 or prefix_ids[0] != self.tokenizer.bos_token_id:
                            token_idx += 1
                # ----------------------------------

                if token_idx < x.shape[1]:
                    start = token_idx
                    end = start + len_repl
                    valid_end = min(end, x.shape[1])
                    len_to_copy = valid_end - start
                    
                    # 1. Insert replacement text
                    if len_to_copy > 0:
                        x[b, start:valid_end] = replacement_tensor[:len_to_copy]
                    
                    # 2. Fill remainder with PAD (stops generation for this sample)
                    if valid_end < x.shape[1]:
                        x[b, valid_end:] = pad_id

    @torch.no_grad()
    def generate(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDAGeneratorConfig | None = None,
        **kwargs
    ) -> GeneratorOutput | torch.Tensor:
        if config is None:
            config = LLaDAGeneratorConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict_in_generate = kwargs.get("return_dict_in_generate", config.return_dict_in_generate)
        repetition_penalty = kwargs.get("repetition_penalty", config.repetition_penalty)

        assert 1 <= block_length
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
        attention_mask = (x != eos_id).long() if B > 1 else None

        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_length)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict_in_generate else None

        for b in range(num_blocks):
            block_mask_index = torch.zeros((B, block_length), dtype=torch.bool, device=x.device)

            for j in range(B):
                start = prompt_lens[j] + b * block_length
                end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                # --- Optimization: Break if no masks left in the entire batch ---
                if not torch.any(x == mask_id):
                    break
                
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                logits = apply_repetition_penalty(
                    logits, x, penalty=repetition_penalty, mask_token_id=mask_id, eos_token_id=eos_id
                )

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_length :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    masks_left = (x[j] == mask_id).sum().item()
                    k = min(k, masks_left)
                    
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

                # ----- CHECK STOP CRITERIA -----
                self._apply_stop_criteria(x)
                # -------------------------------

                if histories is not None:
                    histories.append(x.clone())

        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config,
        **kwargs
    ) -> GeneratorOutput | torch.Tensor:
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict_in_generate = kwargs.get("return_dict_in_generate", config.return_dict_in_generate)
        repetition_penalty = kwargs.get("repetition_penalty", config.repetition_penalty)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_length is None:
            block_length = T

        assert 1 <= block_length
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t
        attention_mask = (x != eos_id).long() if B > 1 else None

        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(T / block_length)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict_in_generate else None

        attention_mask = (x != eos_id).long()

        for b in range(num_blocks):
            start = b * block_length
            stop = min(start + block_length, T)

            block_mask_index = torch.zeros(
                (B, block_length), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                # --- Optimization: Break if no masks left in the entire batch ---
                if not torch.any(x == mask_id):
                    break
                
                mask_index_full = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                logits = apply_repetition_penalty(
                    logits, x, penalty=repetition_penalty, mask_token_id=mask_id, eos_token_id=eos_id
                )

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    end_j = start + widths[j]
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    masks_left = (x[j] == mask_id).sum().item()
                    k = min(k, masks_left)

                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                x[transfer_index] = x0[transfer_index]

                # ----- CHECK STOP CRITERIA -----
                self._apply_stop_criteria(x)
                # -------------------------------
                
                if histories is not None:
                    histories.append(x.clone())

        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)
