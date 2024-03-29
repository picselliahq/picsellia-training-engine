from enum import Enum


class DatasetSplitName(Enum):
    """
    An enumeration of possible dataset split names.

    This enumeration defines standard names for dataset splits commonly used
    in machine learning and data processing workflows. The splits typically
    represent different subsets of the data intended for specific purposes
    such as training, validation, and testing.

    Attributes:
        TRAIN (str): Represents the training split, used for training models.
        VAL (str): Represents the validation split, used for tuning model hyperparameters and preventing overfitting.
        TEST (str): Represents the test split, used for evaluating the final model performance on unseen data.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
