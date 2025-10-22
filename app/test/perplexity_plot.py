import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def plot_heatmap(results: Dict, departments: List[str], path: Path):
	try:
		import matplotlib.pyplot as plt
		import numpy as np
	except ImportError:
		print("matplotlib not available; skipping heatmap")
		return
	labels = [dept.upper() for dept in departments]
	lora_names = list(results["lora_models"].keys())
	if not lora_names:
		return
	data = []
	for lora in lora_names:
		row = []
		for dept in departments:
			value = results["lora_models"][lora].get(dept)
			row.append(value if value is not None else float("nan"))
		data.append(row)
	base = [results["base_model"].get(dept) for dept in departments]
	data = [base] + data
	labels = ["BASE"] + [name.upper() for name in lora_names]
	arr = np.array(data, dtype=float)
	fig, ax = plt.subplots(figsize=(1.5 * len(departments), 0.8 * len(labels) + 1))
	im = ax.imshow(arr, cmap="viridis", aspect="auto")
	ax.set_xticks(range(len(departments)), [dept.upper() for dept in departments])
	ax.set_yticks(range(len(labels)), labels)
	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]):
			value = arr[i, j]
			if not np.isnan(value):
				ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Perplexity")
	ax.set_title("Perplexity Benchmark")
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)


def plot_accuracy(results: Dict, path: Path):
	try:
		import matplotlib.pyplot as plt
		import numpy as np
	except ImportError:
		print("matplotlib not available; skipping accuracy plot")
		return
	names = list(results["accuracy"].keys())
	if not names:
		return
	values = [results["accuracy"].get(name, float("nan")) for name in names]
	fig, ax = plt.subplots(figsize=(1.2 * len(names) + 1, 4))
	ax.bar([name.upper() for name in names], values, color="steelblue")
	ax.set_ylim(0, 1)
	ax.set_ylabel("Proportion of datasets beaten")
	ax.set_title("LoRA Accuracy vs Base (Perplexity Wins)")
	for index, value in enumerate(values):
		if not np.isnan(value):
			ax.text(index, value + 0.02, f"{value:.2f}", ha="center")
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)
