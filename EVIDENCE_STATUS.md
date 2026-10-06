# OpenMythos — Evidence Status

**Date:** 2026-10-06  
**Role:** experimental/theoretical model architecture  
**Production authority:** none

OpenMythos is useful as code and architecture for exploring recurrent-depth / looped-transformer ideas. Its public configuration presets and training scripts should not be read as evidence that every advertised scale has been trained, benchmarked, reproduced, or validated.

## Boundaries

- **Configuration != checkpoint.** A 1B–1T preset describes shapes/hyperparameters; it does not establish trained weights.
- **Training script != completed training run.** A script and dataset target do not establish convergence, final loss, capability, cost, or reproducibility.
- **Context/output fields != demonstrated operating envelope.** Large configured lengths require separate memory, stability, quality, and throughput evidence.
- **Implementation != proprietary reconstruction.** The project is not evidence about Anthropic's private architecture and is not affiliated with Anthropic.
- **Local execution != independent reproduction.** Independent evidence needs a frozen artifact, environment, receipt, and non-origin reproduction.

## Recommended next experiment

Freeze one small configuration, dataset slice, seed set, compute budget, and baseline. Publish training/evaluation receipts and compare recurrent-depth behavior against an equivalent-compute non-recurrent baseline. Keep failed or negative results in the record.
