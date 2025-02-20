# Examples for Picsellia Training Engine

Welcome to the `examples` directory of the **Picsellia Training Engine** repository! This folder contains practical examples to help you build, customize, and deploy your own pipelines, steps, and models using the tools provided by this repository. Whether you're new to Picsellia or an experienced user, these examples will guide you in leveraging the full potential of the training engine.

## Folder Structure

The `examples` folder is divided into subdirectories to organize examples by functionality:

1. `steps/`

Examples of custom steps you can create and integrate into your pipelines. Each example demonstrates how to:

- Define a step using the `@step` decorator.

- Handle inputs and outputs.

- Log relevant information for debugging or monitoring.

2. `pipelines/`

Full pipeline examples that combine multiple steps. These examples include:

- Basic pipelines with sequential steps.

- Advanced pipelines with custom contexts and configurations.

3. `context/`

Examples of creating and using custom contexts for pipelines. These contexts allow you to:

- Store and manage hyperparameters, augmentation settings, and export configurations.

- Share data between different steps in a pipeline.

4. `utils/`

Utility scripts that provide common functionalities such as:

- Dataset splitting and preprocessing.

- Logging and debugging helpers.

## Getting Started

To get started with the examples, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/picselliahq/picsellia-training-engine.git
    cd picsellia-training-engine/examples
    ```

2. **Install dependencies:**
Ensure you have all the required Python dependencies installed. Run the following command from the root of the repository:

    ```bash
    pip install -r requirements-dev.txt
    ```

3. **Explore the examples:**
Navigate to the relevant subdirectory (e.g., steps/, pipelines/) and review the provided scripts. Each script is self-contained and includes comments to help you understand the implementation.

4. **Run an example:**
For example, to run a basic pipeline example:

    ```bash
    python examples/pipelines/basic_pipeline.py
    ```
