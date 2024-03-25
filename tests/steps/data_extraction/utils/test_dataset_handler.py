import pytest


class TestDatasetHandler:
    def test_get_dataset_collection_one_dataset(self, mock_dataset_handler_one_dataset):
        dataset_collection = mock_dataset_handler_one_dataset.get_dataset_collection()
        assert dataset_collection.train.name == "train"
        assert dataset_collection.val.name == "val"
        assert dataset_collection.test.name == "test"

    def test_get_dataset_collection_two_datasets(
        self, mock_dataset_handler_two_datasets
    ):
        dataset_collection = mock_dataset_handler_two_datasets.get_dataset_collection()
        assert dataset_collection.train.name == "train"
        assert dataset_collection.val.name == "val"
        assert dataset_collection.test.name == "test"

    def test_get_dataset_collection_three_datasets(
        self, mock_dataset_handler_three_datasets
    ):
        dataset_collection = (
            mock_dataset_handler_three_datasets.get_dataset_collection()
        )
        assert dataset_collection.train.name == "train"
        assert dataset_collection.val.name == "val"
        assert dataset_collection.test.name == "test"

    def test_get_split_ratios_one_dataset(self, mock_dataset_handler_one_dataset):
        split_ratios = mock_dataset_handler_one_dataset._get_split_ratios(
            count_datasets=1
        )
        assert split_ratios == [0.8, 0.1, 0.1]

    def test_get_split_ratios_two_datasets(self, mock_dataset_handler_two_datasets):
        split_ratios = mock_dataset_handler_two_datasets._get_split_ratios(
            count_datasets=2
        )
        assert split_ratios == [0.8, 0.2]

    def test_get_split_ratios_three_datasets(self, mock_dataset_handler_three_datasets):
        with pytest.raises(RuntimeError):
            mock_dataset_handler_three_datasets._get_split_ratios(count_datasets=3)

    def test_handle_one_dataset(
        self, mock_dataset_handler_one_dataset, mock_train_dataset_version
    ):
        dataset_collection = mock_dataset_handler_one_dataset._handle_one_dataset(
            train_dataset_version=mock_train_dataset_version
        )
        assert dataset_collection.train.name == "train"
        assert dataset_collection.val.name == "val"
        assert dataset_collection.test.name == "test"

        # check that val assets are 10% of the train assets in train_dataset_version
        assert len(dataset_collection.train.multi_asset) == 8
        assert len(dataset_collection.val.multi_asset) == 1
        assert len(dataset_collection.test.multi_asset) == 1

        # check that the labelmap is the same in all contexts
        assert dataset_collection.train.labelmap == dataset_collection.val.labelmap
        assert dataset_collection.train.labelmap == dataset_collection.test.labelmap

        # check that the labelmap is the same in all contexts
        assert dataset_collection.train.labelmap == dataset_collection.val.labelmap
        assert dataset_collection.train.labelmap == dataset_collection.test.labelmap

        # check that train assets are in train_dataset_version
        for asset in dataset_collection.train.multi_asset:
            assert asset in dataset_collection.train.dataset_version.list_assets()

        # check that val assets are in train_dataset_version
        for asset in dataset_collection.val.multi_asset:
            assert asset in dataset_collection.train.dataset_version.list_assets()

        # check that test assets are in train_dataset_version
        for asset in dataset_collection.test.multi_asset:
            assert asset in dataset_collection.train.dataset_version.list_assets()

    def test_handle_two_datasets(
        self,
        mock_dataset_handler_two_datasets,
        mock_train_dataset_version,
        mock_test_dataset_version,
    ):
        dataset_collection = mock_dataset_handler_two_datasets._handle_two_datasets(
            train_dataset_version=mock_train_dataset_version,
            test_dataset_version=mock_test_dataset_version,
        )
        assert dataset_collection.train.name == "train"
        assert dataset_collection.val.name == "val"
        assert dataset_collection.test.name == "test"

        # check that val assets are 10% of the train assets in train_dataset_version
        assert len(dataset_collection.train.multi_asset) == 8
        assert len(dataset_collection.val.multi_asset) == 2
        assert len(dataset_collection.test.multi_asset) == 10

        # check that the labelmap is the same in all contexts
        assert dataset_collection.train.labelmap == dataset_collection.val.labelmap

        # check that the dataset version is the same in all contexts
        assert (
            dataset_collection.train.dataset_version
            == dataset_collection.val.dataset_version
        )

        # check that train assets are in train_dataset_version
        for asset in dataset_collection.train.multi_asset:
            assert asset in dataset_collection.train.dataset_version.list_assets()

        # check that val assets are in train_dataset_version
        for asset in dataset_collection.val.multi_asset:
            assert asset in dataset_collection.train.dataset_version.list_assets()

        # check that test assets are in train_dataset_version
        for asset in dataset_collection.test.multi_asset:
            assert asset in dataset_collection.test.dataset_version.list_assets()

    def test_handle_three_datasets(
        self,
        mock_dataset_handler_three_datasets,
        mock_train_dataset_version,
        mock_val_dataset_version,
        mock_test_dataset_version,
    ):
        dataset_collection = mock_dataset_handler_three_datasets._handle_three_datasets(
            train_dataset_version=mock_train_dataset_version,
            val_dataset_version=mock_val_dataset_version,
            test_dataset_version=mock_test_dataset_version,
        )
        assert dataset_collection.train.name == "train"
        assert dataset_collection.val.name == "val"
        assert dataset_collection.test.name == "test"

        # check that val assets are 10% of the train assets in train_dataset_version
        assert len(dataset_collection.train.multi_asset) == 10
        assert len(dataset_collection.val.multi_asset) == 10
        assert len(dataset_collection.test.multi_asset) == 10

        # check that train assets are in train_dataset_version
        for asset in dataset_collection.train.multi_asset:
            assert asset in dataset_collection.train.dataset_version.list_assets()

        # check that val assets are in train_dataset_version
        for asset in dataset_collection.val.multi_asset:
            assert asset in dataset_collection.val.dataset_version.list_assets()

        # check that test assets are in train_dataset_version
        for asset in dataset_collection.test.multi_asset:
            assert asset in dataset_collection.test.dataset_version.list_assets()
