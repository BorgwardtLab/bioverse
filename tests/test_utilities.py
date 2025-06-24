import json
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import awkward as ak
import numpy as np
import pytest
import yaml

from bioverse.utilities import *


# Array utilities tests
class TestArrayUtilities:

    def test_flatten(self):
        data = ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        # Test default (flatten all axes)
        result = flatten(data)
        expected = ak.Array([1, 2, 3, 4, 5, 6, 7, 8])
        assert ak.all(result == expected)

        # Test with specific axis
        result = flatten(data, 1)
        expected = ak.Array([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert ak.all(result == expected)

        # Test with exclude
        result = flatten(data, exclude=1)
        expected = ak.Array([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert ak.all(result == expected)

    def test_onehot(self):
        data = ak.Array([0, 1, 2])
        result = onehot(data, 3)
        expected = ak.Array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert ak.all(result == expected)

    def test_cumsum(self):
        data = ak.Array([[1, 2, 3], [4, 5]])
        result = cumsum(data)
        expected = ak.Array([[1, 3, 6], [4, 9]])
        assert ak.all(result == expected)

    def test_diff(self):
        data = ak.Array([[1, 2, 3], [4, 5]])
        result = diff(data)
        expected = ak.Array([[1, 1], [1]])
        assert ak.all(result == expected)


# IO utilities tests
class TestIOUtilities:
    def test_save_load_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            data = {"test": "data"}
            save(data, temp.name)
            loaded = load(temp.name)
            assert loaded == data
            os.unlink(temp.name)

    def test_save_load_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
            data = {"test": "data"}
            save(data, temp.name)
            loaded = load(temp.name)
            assert loaded == data
            os.unlink(temp.name)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load("nonexistent_file.json")

    def test_load_with_default(self):
        default_value = {"default": "value"}
        result = load("nonexistent_file.json", default=default_value)
        assert result == default_value

    def test_warn_info_note(self):
        with patch("bioverse.utilities.io.console.print") as mock_print:
            # Test warn function
            warn("test warning")
            mock_print.assert_called_with("Warning: test warning", style="yellow")

            # Test info function
            info("test info")
            mock_print.assert_called_with("Info: test info", style="cyan")

            # Test note function
            note("test note")
            mock_print.assert_called_with("Note: test note", style="#666666")

    def test_zip_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(b"test data")
            temp.close()

            zipped = zip_file(temp.name)
            assert zipped.exists()
            assert zipped.suffix == ".gz"

            # Clean up
            os.unlink(temp.name)
            os.unlink(zipped)

    def test_interleave(self):
        result = interleave([1, 2], [3, 4], [5, 6])
        assert result == [1, 3, 5, 2, 4, 6]

    def test_long_to_wide(self):
        data = [
            {"id": 1, "feature": "color", "value": "red"},
            {"id": 1, "feature": "size", "value": "large"},
            {"id": 2, "feature": "color", "value": "blue"},
            {"id": 2, "feature": "size", "value": "small"},
        ]
        result = long_to_wide(data, "id", "feature", "value")
        expected = [
            {"id": 1, "color": "red", "size": "large"},
            {"id": 2, "color": "blue", "size": "small"},
        ]
        # Sort to ensure consistent comparison
        result = sorted(result, key=lambda x: x["id"])
        expected = sorted(expected, key=lambda x: x["id"])
        assert result == expected

    @pytest.mark.skip("Requires proper molecule array structure")
    def test_molecule_to_pdb(self):
        # Would need to mock a proper molecule structure
        pass

    @patch("bioverse.utilities.io.Progress")
    def test_progressbar(self, mock_progress):
        mock_task = 1
        mock_progress_instance = mock_progress.return_value
        mock_progress_instance.add_task.return_value = mock_task

        data = [1, 2, 3]
        result = list(progressbar(data, description="Test"))

        mock_progress.assert_called_once()
        mock_progress_instance.add_task.assert_called_with(description="Test", total=3)
        assert len(mock_progress_instance.update.mock_calls) == 3
        assert result == data

    def test_itlen(self):
        # Test with sequence
        data = [1, 2, 3]
        length, result = itlen(data)
        assert length == 3
        assert result == data

        # Test with iterator
        data_iter = iter([1, 2, 3])
        with patch("bioverse.utilities.io.length", return_value=3):
            length, result = itlen(data_iter)
            assert length == 3
            assert list(result) == [1, 2, 3]

    def test_length(self):
        data = [1, 2, 3]
        assert length(data) == 3

        data_iter = iter([1, 2, 3])
        assert length(data_iter) == 3

    @patch("bioverse.utilities.io.os")
    def test_glob_delete(self, mock_os):
        with patch(
            "bioverse.utilities.io.glob.iglob", return_value=["file1.txt", "file2.txt"]
        ):
            glob_delete("*.txt")
            assert mock_os.remove.call_count == 2
            mock_os.remove.assert_any_call("file1.txt")
            mock_os.remove.assert_any_call("file2.txt")

    @patch("bioverse.utilities.io.os.makedirs")
    @patch("bioverse.utilities.io.open", new_callable=mock_open)
    @patch("bioverse.utilities.io.requests.get")
    def test_download(self, mock_get, mock_open, mock_makedirs):
        # Set up the mock response
        mock_response = mock_get.return_value
        mock_response.headers = {"content-length": "1024"}
        mock_response.raise_for_status = lambda: None
        mock_response.iter_content.return_value = [b"data"]

        with patch("bioverse.utilities.io.Path.exists", return_value=False):
            with patch(
                "bioverse.utilities.io.progressbar", side_effect=lambda x, **kwargs: x
            ):
                # Mock Path class behavior
                with patch("bioverse.utilities.io.Path") as mock_path:
                    # Configure the mock path
                    mock_path_instance = mock_path.return_value
                    mock_path_instance.exists.return_value = False
                    mock_path_instance.parent = Path(".")
                    mock_path_instance.name = "output.txt"
                    mock_path_instance.suffix = ".txt"
                    mock_path_instance.stem = "output"
                    mock_path_instance.with_suffix.return_value = Path("output.txt")
                    mock_path_instance.__str__.return_value = "output.txt"

                    # Configure the mock Path call for the url_path
                    mock_path.side_effect = lambda p, *args, **kwargs: (
                        Path(p)
                        if isinstance(p, str) and p.startswith("http")
                        else mock_path_instance
                    )

                    download("http://example.com/file.txt", "output.txt")

                    mock_get.assert_called_with(
                        "http://example.com/file.txt",
                        stream=True,
                        headers={"User-Agent": "XY"},
                    )
                    mock_makedirs.assert_called_once()
                    mock_open.assert_called_once()

    @patch("bioverse.utilities.io.os")
    @patch("bioverse.utilities.io.tarfile.open")
    def test_extract(self, mock_tarfile_open, mock_os):
        mock_tar_file = mock_tarfile_open.return_value.__enter__.return_value
        mock_member1 = type("obj", (object,), {"path": "dir/file1.txt"})
        mock_member2 = type("obj", (object,), {"path": "dir/file2.txt"})
        mock_tar_file.getmembers.return_value = [mock_member1, mock_member2]

        with patch(
            "bioverse.utilities.io.progressbar", side_effect=lambda x, **kwargs: x
        ):
            extract("archive.tar", "output_dir")

            mock_tarfile_open.assert_called_with(Path("archive.tar"), "r")
            assert mock_tar_file.extract.call_count == 2
            mock_os.remove.assert_called_once_with(Path("archive.tar"))

    @patch("bioverse.utilities.io.Parallel")
    def test_parallelize(self, mock_parallel):
        mock_parallel_instance = mock_parallel.return_value
        mock_parallel_instance.return_value = iter([2, 4, 6])

        with patch("bioverse.utilities.io.config.workers", 4):
            with patch("bioverse.utilities.io.Progress"):
                with patch("bioverse.utilities.io.itlen", return_value=(3, [1, 2, 3])):
                    result = list(
                        parallelize(lambda x: x * 2, [1, 2, 3], progress=True)
                    )
                    assert result == [2, 4, 6]
                    mock_parallel.assert_called_with(n_jobs=4, return_as="generator")

    @pytest.mark.skip("Requires proper Batch class mock")
    def test_batched_rebatch_save_shards(self):
        # These functions depend on the Batch class implementation
        pass

    @patch("bioverse.utilities.io.sys.modules")
    def test_alias(self, mock_modules):
        # Create a mock module
        mock_module = type("module", (object,), {})
        mock_modules.__getitem__.return_value = mock_module

        # Define a test class and decorate it
        class TestClass:
            pass

        decorated = alias("AliasName")(TestClass)

        # Check that the alias was set on the module
        assert hasattr(mock_module, "AliasName")
        assert getattr(mock_module, "AliasName") == TestClass
        assert decorated == TestClass  # Decorator should return the original class


# Data utilities tests
@pytest.mark.skip("Requires proper Batch class mock")
class TestDataUtilities:
    def test_featurize(self):
        # This test is skipped as it requires a proper Batch class mock
        pass

    def test_set_resolution(self):
        # This test is skipped as it requires a proper Batch class mock
        pass


# Factory utilities tests
@pytest.mark.skip("Requires more complex mocking of dependencies")
class TestFactoryUtilities:
    def test_benchmark_factory(self):
        # This would require complex mocking of the benchmark and related classes
        pass


if __name__ == "__main__":
    pytest.main()
