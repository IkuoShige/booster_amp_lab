from __future__ import annotations

import torch


class TrackAdapterWorldModelReplayBuffer:
    """Persistent replay buffer for Track Adapter dynamics-model samples."""

    def __init__(
        self,
        history_shape,
        reference_shape,
        target_shape,
        valid_mask_shape,
        action_sequence_shape,
        buffer_size,
        sample_device,
        storage_device="cpu",
        storage_dtype=torch.float16,
    ):
        self.history_shape = tuple(history_shape)
        self.reference_shape = tuple(reference_shape)
        self.target_shape = tuple(target_shape)
        self.valid_mask_shape = tuple(valid_mask_shape)
        self.action_sequence_shape = tuple(action_sequence_shape)
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError(f"TrackAdapterWorldModelReplayBuffer buffer_size must be positive, got {buffer_size}.")
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.storage_dtype = storage_dtype

        self.history = torch.zeros(
            self.buffer_size, *self.history_shape, device=self.storage_device, dtype=self.storage_dtype
        )
        self.references = torch.zeros(
            self.buffer_size, *self.reference_shape, device=self.storage_device, dtype=self.storage_dtype
        )
        self.targets = torch.zeros(
            self.buffer_size, *self.target_shape, device=self.storage_device, dtype=self.storage_dtype
        )
        self.valid_masks = torch.zeros(
            self.buffer_size, *self.valid_mask_shape, device=self.storage_device, dtype=self.storage_dtype
        )
        self.action_sequences = torch.zeros(
            self.buffer_size, *self.action_sequence_shape, device=self.storage_device, dtype=self.storage_dtype
        )
        self.step = 0
        self.num_samples = 0
        self.total_inserted = 0

    @staticmethod
    def _flatten_samples(tensor, sample_shape):
        return tensor.detach().reshape(-1, *tuple(sample_shape))

    @staticmethod
    def _finite_rows(tensor):
        return torch.isfinite(tensor.reshape(tensor.shape[0], -1)).all(dim=1)

    @staticmethod
    def _dtype_from_name(name):
        if isinstance(name, torch.dtype):
            return name
        normalized = str(name).lower()
        if normalized in ("float16", "half", "fp16"):
            return torch.float16
        if normalized in ("bfloat16", "bf16"):
            return torch.bfloat16
        if normalized in ("float32", "float", "fp32"):
            return torch.float32
        raise ValueError(f"Unsupported Track Adapter world-model replay dtype: {name}")

    def insert(self, history, references, targets, valid_masks, action_sequences):
        history = self._flatten_samples(history, self.history_shape)
        references = self._flatten_samples(references, self.reference_shape)
        targets = self._flatten_samples(targets, self.target_shape)
        valid_masks = self._flatten_samples(valid_masks, self.valid_mask_shape)
        action_sequences = self._flatten_samples(action_sequences, self.action_sequence_shape)

        if not (
            history.shape[0]
            == references.shape[0]
            == targets.shape[0]
            == valid_masks.shape[0]
            == action_sequences.shape[0]
        ):
            raise ValueError("Track Adapter world-model replay samples have inconsistent leading dimensions.")

        valid_rows = valid_masks.reshape(valid_masks.shape[0], -1).sum(dim=1) > 0
        valid_rows = (
            valid_rows
            & self._finite_rows(history)
            & self._finite_rows(references)
            & self._finite_rows(targets)
            & self._finite_rows(valid_masks)
            & self._finite_rows(action_sequences)
        )
        if not bool(valid_rows.any()):
            return 0

        history = history[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        references = references[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        targets = targets[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        valid_masks = valid_masks[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        action_sequences = action_sequences[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)

        num_samples = history.shape[0]
        if num_samples >= self.buffer_size:
            history = history[-self.buffer_size :]
            references = references[-self.buffer_size :]
            targets = targets[-self.buffer_size :]
            valid_masks = valid_masks[-self.buffer_size :]
            action_sequences = action_sequences[-self.buffer_size :]
            num_samples = self.buffer_size

        end_idx = self.step + num_samples
        if end_idx <= self.buffer_size:
            self.history[self.step : end_idx].copy_(history)
            self.references[self.step : end_idx].copy_(references)
            self.targets[self.step : end_idx].copy_(targets)
            self.valid_masks[self.step : end_idx].copy_(valid_masks)
            self.action_sequences[self.step : end_idx].copy_(action_sequences)
        else:
            first = self.buffer_size - self.step
            second = end_idx - self.buffer_size
            self.history[self.step :].copy_(history[:first])
            self.references[self.step :].copy_(references[:first])
            self.targets[self.step :].copy_(targets[:first])
            self.valid_masks[self.step :].copy_(valid_masks[:first])
            self.action_sequences[self.step :].copy_(action_sequences[:first])
            self.history[:second].copy_(history[first:])
            self.references[:second].copy_(references[first:])
            self.targets[:second].copy_(targets[first:])
            self.valid_masks[:second].copy_(valid_masks[first:])
            self.action_sequences[:second].copy_(action_sequences[first:])

        self.step = (self.step + num_samples) % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + num_samples)
        self.total_inserted += num_samples
        return num_samples

    def sample(self, batch_size):
        if self.num_samples <= 0:
            raise RuntimeError("Cannot sample an empty Track Adapter world-model replay buffer.")
        batch_size = max(1, min(int(batch_size), self.num_samples))
        indices = torch.randint(self.num_samples, (batch_size,), device=self.storage_device)
        return (
            self.history[indices].to(device=self.sample_device, dtype=torch.float32),
            self.references[indices].to(device=self.sample_device, dtype=torch.float32),
            self.targets[indices].to(device=self.sample_device, dtype=torch.float32),
            self.valid_masks[indices].to(device=self.sample_device, dtype=torch.float32),
            self.action_sequences[indices].to(device=self.sample_device, dtype=torch.float32),
        )

    def feed_forward_generator(self, num_mini_batches, mini_batch_size):
        for _ in range(int(num_mini_batches)):
            yield self.sample(mini_batch_size)
