from src.models.dataset.dataset_context import DatasetContext


class DatasetCollection:
    """
    A collection of dataset contexts for different splits of a dataset.

    This class aggregates dataset contexts for the common splits used in machine learning projects:
    training, validation, and testing. It provides a convenient way to access and manipulate these
    dataset contexts as a unified object. The class supports direct access to individual dataset
    contexts, iteration over all contexts, and collective operations on all contexts, such as downloading
    assets.

    Attributes:
        train (DatasetContext): The dataset context for the training split.
        val (DatasetContext): The dataset context for the validation split.
        test (DatasetContext): The dataset context for the testing split.
    """

    def __init__(
        self,
        train_dataset_context: DatasetContext,
        val_dataset_context: DatasetContext,
        test_dataset_context: DatasetContext,
    ):
        """
        Initializes a new DatasetCollection with specified dataset contexts for training,
        validation, and testing splits.

        Args:
            train_dataset_context (DatasetContext): The dataset context for the training split.
            val_dataset_context (DatasetContext): The dataset context for the validation split.
            test_dataset_context (DatasetContext): The dataset context for the testing split.
        """
        self.train = train_dataset_context
        self.val = val_dataset_context
        self.test = test_dataset_context

    def __getitem__(self, key):
        """
        Allows direct access to a dataset context via its split name.

        Args:
            key (str): The split name ('train', 'val', or 'test') of the dataset context to access.

        Returns:
            DatasetContext: The dataset context associated with the given split name.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        Allows setting a dataset context for a specific split via its split name.

        Args:
            key (str): The split name ('train', 'val', or 'test') for which to set the dataset context.
            value (DatasetContext): The dataset context to set for the given split.
        """
        setattr(self, key, value)

    def __iter__(self):
        """
        Enables iteration over the dataset contexts in the collection.

        Returns:
            iterator: An iterator over the dataset contexts (train, val, test) in the collection.
        """
        return iter([self.train, self.val, self.test])

    def download(self):
        """
        Downloads the assets and COCO files for all dataset contexts in the collection.

        Iterates over each dataset context in the collection and invokes its methods
        to download assets and COCO files. This is a collective operation that appliess
        to the training, validation, and testing splits.
        """
        for dataset_context in self:
            dataset_context.download()
