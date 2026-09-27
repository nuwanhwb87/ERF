# ERF

Extreme Rainfall Forecasting (ERF) using diffusion-based deep generative models.

## Objective

This repository defines a two-stage forecasting roadmap for extreme rainfall prediction:

1. **Initial stage:** develop a new architecture for **deterministic forecasting**
2. **Second stage:** extend the same architecture for **ensemble forecasting**

## Proposed Architecture

### Stage 1 — Deterministic Forecasting

Build a deterministic backbone that learns the spatiotemporal structure of extreme rainfall events from historical atmospheric and precipitation inputs.

Core design goals:

- capture multi-scale spatial rainfall patterns
- model temporal evolution of extreme events
- preserve physically consistent forecast structure
- provide a strong backbone for later probabilistic sampling

Suggested model blocks:

- **Encoder:** extracts multi-scale features from meteorological inputs
- **Temporal module:** models sequence evolution across forecast lead times
- **Decoder:** reconstructs high-resolution deterministic rainfall forecasts
- **Training objective:** regression-focused loss for accurate extreme rainfall prediction

### Stage 2 — Ensemble Forecasting

Reuse the deterministic backbone as the conditioning architecture for diffusion-based ensemble generation.

Stage 2 goals:

- generate multiple plausible rainfall futures from the same initial condition
- represent forecast uncertainty for extreme events
- preserve deterministic skill while improving probabilistic coverage

Suggested extension path:

- use the Stage 1 deterministic network as the conditioning pathway
- add a diffusion-based generative head for stochastic forecast sampling
- produce an ensemble by repeated denoising/sampling conditioned on the deterministic features

## Delivery Plan

### Initial Stage Output

- deterministic extreme rainfall forecasting architecture
- training and evaluation pipeline for single best-estimate forecasts
- benchmark metrics for heavy/extreme rainfall events

### Second Stage Output

- diffusion-based ensemble forecasting extension
- uncertainty-aware rainfall forecast generation
- ensemble verification for reliability, spread, and extreme-event capture

## Repository Status

The repository is currently in the initial planning stage. The next implementation step is to add the deterministic forecasting model and training pipeline, then extend that architecture into a diffusion-based ensemble forecaster.
