# Perplexity Benchmarking for LoRA Models

This directory contains scripts for evaluating the performance of LoRA-fine-tuned models on department-specific datasets using perplexity as the metric.

## Overview

The `perplexity.py` script performs a comprehensive benchmark:

- Loads the base Qwen model and each department LoRA (finance, hr, it_support, engineering)
- Evaluates perplexity on the training datasets for each department
- Compares LoRA performance against the base model
- Generates visualizations and accuracy metrics
## Usage

Run the benchmark from the project root:

```bash
env/bin/python -m app.test.perplexity \
  --max-samples 50 \
  --heatmap outputs/perplexity_heatmap.png \
  --accuracy-plot outputs/perplexity_accuracy.png
```
