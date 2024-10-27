# SPSA Gradient Estimator for Network Model

This repository provides an implementation of the Simultaneous Perturbation Stochastic Approximation (SPSA) gradient estimation method. This is a method used to approximate gradients for model optimization in scenarios where direct gradient computation is impractical, particularly in high-dimensional settings. The SPSA method efficiently estimates gradients by applying random perturbations, enabling robust updates even when traditional gradient methods are infeasible or costly.

## Features

- **SPSA Gradient Estimation**: Efficiently estimate gradients using simultaneous perturbation.
- **Generalizable**: Can be applied to various models for optimization in noisy or complex environments.
- **Configurable Averaging and Parameters**: Allows for tuning hyperparameters to improve estimation accuracy.

## Requirements

- Python 3.6+
- PyTorch
- CUDA (for GPU acceleration)

## Installation

To install the necessary packages, run:

```bash
pip install torch
```

## Usage

This class is designed to be used as part of a PyTorch training loop to estimate gradients. Below are explanations of the primary components and how to use the `SPSA` class.

### Code Overview

The `SPSA` class has the following structure:

- **Initialization** (`__init__`): Initializes the SPSA estimator with model parameters and a loss criterion. Configurable parameters include:
  - `sp_avg`: Number of samples for averaging gradient estimates.
  - `b1`: Momentum term coefficient.
  - `o`, `c`, `a`, `alpha`, `gamma`: Coefficients used to calculate the perturbation and learning rates.

- **Gradient Estimation** (`estimate`): Calculates the gradient estimate using SPSA.
  - **Inputs**:
    - `epoch`: The current epoch number.
    - `images`: Batch of input images.
    - `labels`: True labels for the input images.
  - **Outputs**:
    - Returns an updated model with adjusted parameters based on the gradient estimation.
  
### Example

1. **Define Your Model and Criterion**:
    ```python
    import torch.nn as nn

    model = MyNetwork()  # replace with your model
    criterion = nn.CrossEntropyLoss()  # or your chosen loss function
    ```

2. **Initialize SPSA**:
    ```python
    spsa = SPSA(model=model, criterion=criterion)
    ```

3. **Estimate Gradient and Update Parameters**:
    In your training loop, call `spsa.estimate()` to calculate and apply the gradient estimate:
    ```python
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            spsa.estimate(epoch, images, labels)
    ```

## Parameters and Configuration

The SPSA algorithm includes parameters that control the gradient estimation process:

- **`sp_avg`**: The number of samples to average for the gradient estimate.
- **`b1`**: The momentum term to smooth the estimated gradient.
- **`c`, `a`, `alpha`, `gamma`**: Parameters that control the scale and decay of the step size and perturbation.

### Customizing Parameters

You can experiment with these parameters to improve the gradient estimation accuracy for your specific model and dataset. Adjusting the parameters may help achieve better results depending on the complexity of your model.

## Example Configuration

Here is an example configuration with default parameters:
```python
sp_avg = 5
b1 = 0.9
o = 1.0
c = 0.005
a = 0.01
alpha = 0.4
gamma = 0.2
```
