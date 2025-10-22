import gc
import os
import sys
from typing import Any

import torch

from inference import _MODEL_CACHE, _TOKENIZER_CACHE


def _safe_call(target: Any, *args: Any) -> None:
	if not callable(target):
		return
	try:
		target(*args)
	except TypeError:
		try:
			target()
		except Exception:
			pass
	except Exception:
		pass


def _release_model(model: Any) -> None:
	if model is None:
		return
	_safe_call(getattr(model, "to", None), "cpu")
	_safe_call(getattr(model, "cpu", None))
	_safe_call(getattr(model, "detach", None))


def cleanup_loaded_models() -> None:
	for key in list(_MODEL_CACHE.keys()):
		model = _MODEL_CACHE.pop(key, None)
		_release_model(model)
	_TOKENIZER_CACHE.clear()
	gc.collect()
	if torch.cuda.is_available():
		try:
			torch.cuda.empty_cache()
		except Exception:
			pass
		ipc_collect = getattr(torch.cuda, "ipc_collect", None)
		if callable(ipc_collect):
			try:
				ipc_collect()
			except Exception:
				pass
		try:
			torch.cuda.reset_peak_memory_stats()
		except Exception:
			pass


__all__ = ["cleanup_loaded_models"]


if __name__ == "__main__":
    cleanup_loaded_models()
    print("GPU cleanup completed.")
