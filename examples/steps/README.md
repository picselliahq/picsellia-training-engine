# Steps for Picsellia Training Engine

Welcome to the `steps` subdirectory of the Picsellia Training Engine examples! This folder provides reusable step implementations designed to simplify the development and deployment of machine learning pipelines. These steps can be used independently or as part of a pipeline, making it easier to focus on high-level functionality without directly interacting with low-level SDK operations.

## Overview

Steps in the Picsellia Training Engine are modular functions, decorated with `@step`, that encapsulate specific tasks such as data extraction, validation, training, and evaluation. They offer several advantages:

- **Reusability**: Steps can be plugged into different pipelines or run independently.

- **Abstraction**: Hide low-level SDK details, enabling users to focus on their specific logic.

- **Modularity**: Each step is self-contained and can be tested, debugged, or modified individually.

## Included Steps

This directory contains several prebuilt steps, including:

1. **Data Extraction**

- `get_training_dataset_collection`: Extracts datasets from a Picsellia experiment and prepares them for training, validation, and testing. Logs label maps and object distributions for better analysis.

2. **Data Validation**

- `validate_training_data`: Validates a dataset collection, ensuring images are properly extracted, formatted, and free of corruption.

- `validate_processing_data`: Validates individual dataset contexts for processing.

3. **Model Context Management**

- `get_training_model_context`: Extracts model context from a Picsellia experiment, downloads pretrained weights, and initializes the model configuration.

4. **Model Evaluation**

- `evaluate_ultralytics_model_context`: Performs evaluation of an Ultralytics model on a test dataset, processing data in batches and logging results.

## Getting Started with Steps

### Importing and Using Steps

Each step is defined as a Python function decorated with @step. You can import and use these steps in your pipeline or as standalone functions:

```python
from steps.training_data_extractor import get_training_dataset_collection
from steps.data_validator import validate_training_data

# Example usage:
dataset_collection = get_training_dataset_collection()
validate_training_data(dataset_collection)
```

### Running Steps Independently

Since steps are self-contained, you can run them independently for testing or debugging:

```bash
python steps/training_data_extractor.py
```

### Creating Custom Steps

You can create your own steps by decorating a Python function with @step. For example:

```python
from src import step

@step(name="Custom Step")
def custom_step(param):
    print(f"Processing parameter: {param}")
```
