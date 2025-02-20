# Decorators Module

This module contains the implementation of the `@pipeline` decorator, designed to orchestrate and execute processing pipelines in a structured and extensible manner. It provides tools for managing steps, tracking execution states, and handling logging effectively.

## Introduction

The `@pipeline` decorator enables the creation and management of processing pipelines. It allows developers to define individual steps that are automatically registered and executed in sequence within a shared context.

Key benefits of using this module include:

- Clear separation of pipeline stages.
- Context sharing between steps.
- Automatic state tracking for individual steps and the pipeline as a whole.
- Centralized logging for better observability and debugging.

## Pipeline Overview

The core of the module is the `@pipeline` decorator, which wraps a main function to define and execute a sequence of steps. Each step is decorated with `@step`, enabling automatic registration and state management.

### Features

- **Shared Context**: A dictionary-based context is shared across all steps, allowing seamless data passing.
- **Logging**: Automatically logs pipeline execution, with options to save logs to disk or delete them after completion.
- **Step Management**: Steps are dynamically registered and tracked in the `Pipeline.STEPS_REGISTRY`.
- **State Tracking**: Tracks step statuses (`PENDING`, `RUNNING`, `SUCCESS`, etc.) and aggregates them into a pipeline-level status.

## Usage Example

### Defining a Pipeline

```python
from decorators.pipeline import pipeline

@pipeline(
    context={"input_data": "data.csv", "config": "config.yaml"},
    name="DataProcessingPipeline",
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def main_pipeline():
    load_data()
    process_data()
    save_results()

main_pipeline()
```

### Defining Steps

```python
from decorators.steps import step

@step(name="LoadData")
def load_data():
    print("Loading data...")

@step(name="ProcessData")
def process_data():
    print("Processing data...")

@step(name="SaveResults")
def save_results():
    print("Saving results...")
```
