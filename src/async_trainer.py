"""Compatibility export; SkyRL v0.3 contains the former CodePin patches."""

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

CustomFullyAsyncRayPPOTrainer = FullyAsyncRayPPOTrainer

__all__ = ["CustomFullyAsyncRayPPOTrainer"]
