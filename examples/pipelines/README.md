# Pipelines for Picsellia Training Engine

Welcome to the `pipelines` subdirectory of the **Picsellia Training Engine** examples! This directory demonstrates how to build, execute, and customize machine learning pipelines. These pipelines integrate multiple reusable steps, enabling efficient model training, evaluation, and processing workflows.

## Overview

Pipelines in the Picsellia Training Engine:

- Are defined using the `@pipeline` decorator.
- Combine modular steps into a single workflow.
- Simplify complex ML processes into manageable tasks.
- Leverage **parameters** and **contexts** to adapt pipelines to specific needs.

## Key Concepts

### Parameters
Parameters define the configuration for various stages of a pipeline, such as:

- **Hyperparameters:** Control the behavior of model training (e.g., learning rate, batch size).
- **Augmentation Parameters:** Specify data augmentation techniques (e.g., flipping, scaling).
- **Export Parameters:** Define how models or outputs are saved and exported.

These parameters are managed using dedicated classes, such as:
- `UltralyticsHyperParameters`
- `UltralyticsAugmentationParameters`

Parameters are passed to the pipeline context, ensuring a consistent configuration across steps.

### Context
The **context** provides shared state and configuration for a pipeline. It includes:

- Experiment information (e.g., dataset versions, model versions).
- Parameter objects (hyperparameters, augmentation parameters, export parameters).

Examples of context classes include:
- `PicselliaTrainingContext`: For training pipelines.
- `PicselliaProcessingContext`: For processing pipelines.

A context ensures that all steps in a pipeline have access to the necessary resources and configurations.

## Included Pipelines

### 1. **YOLOv8 Classification Training Pipeline**

This pipeline demonstrates a full training workflow for a YOLOv8 classification model. It includes:

- Data extraction and preparation.
- Model training.
- Model evaluation and export.

### 2. **Bounding Box Cropper Processing Pipeline**

This pipeline processes datasets by cropping bounding boxes. It demonstrates:

- Dataset validation.
- Processing workflows for generating cropped datasets.
- Uploading the resulting datasets.

### 3. **PaddleOCR Training Pipeline**

This pipeline is designed for OCR tasks, specifically for training PaddleOCR models. It features:

- Dataset extraction and preparation.
- Model initialization, training, and export.
- Post-training evaluation.

## Getting Started with Pipelines

### Importing and Running Pipelines

Each pipeline is a Python function decorated with `@pipeline`. You can import and execute these pipelines directly:

```python
from pipelines.training_pipeline import yolov8_classification_training_pipeline

yolov8_classification_training_pipeline()
```

### Customizing Pipelines

To adapt a pipeline to your needs, you can:

1. Add or remove steps.
2. Modify step arguments and configurations.
3. Change the pipeline’s context (e.g., hyperparameters, data splits).

For example, to customize a pipeline:

```python
from pipelines.training_pipeline import yolov8_classification_training_pipeline

# Modify hyperparameters
pipeline_context = yolov8_classification_training_pipeline.get_context()
pipeline_context.hyperparameters.learning_rate = 0.01

# Run the customized pipeline
yolov8_classification_training_pipeline(context=pipeline_context)
```

## Creating New Pipelines

To create a new pipeline, follow these steps:

1. **Define a Context:** Define the hyperparameters and configurations your pipeline will use.

   ```python
   def get_context():
       return PicselliaTrainingContext(
           hyperparameters_cls=YourHyperParameters,
           augmentation_parameters_cls=YourAugmentationParameters,
           export_parameters_cls=YourExportParameters,
       )
   ```

2. **Decorate Your Function:** Use `@pipeline` to define your workflow.

   ```python
   @pipeline(context=get_context(), log_folder_path="logs/")
   def my_custom_pipeline():
       # Add your steps here
       pass
   ```

3. **Integrate Steps:** Combine existing or custom steps to build your pipeline logic.

## Contributing New Pipelines

We welcome contributions to the pipeline examples! To contribute:

1. Write clear docstrings for your pipeline.
2. Test the pipeline thoroughly.
3. Organize your pipeline in a dedicated file with meaningful naming.

## Need Help?

For more details about pipelines or other components of the Picsellia Training Engine, check the main repository's documentation or open an issue in the [issue tracker](https://github.com/your-repo/picsellia-training-engine/issues).

---

Happy coding!

**The Picsellia Team**
