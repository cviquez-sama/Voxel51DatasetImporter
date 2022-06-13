from __future__ import annotations

import json
import os
import unittest
from unittest.mock import mock_open, patch

import fiftyone as fo
from pytest_mock_resources import create_mongo_fixture

from .context import CustomLabeledImageDataset, SAMADatasetImporter

DATA = json.dumps([{
    "id": "001",
    "data": {
        "Name": "asset01.jpg",
        "Image": "https://asset.samasource.org",
        "Annotation Height": "720",
        "Annotation Width": "1280"
    },
    "answers": {
        "Image Annotation": {
            "layers": {
                "vector_tagging": [
                    {
                        "shapes": [
                            {
                                "tags": {"Vehicle": "other_vehicle"},
                                "type": "rectangle",
                                "index": 1,
                                "points": [
                                  [67, 199],
                                  [254, 199],
                                  [67, 433],
                                  [254, 433]
                                ]
                            }
                        ],
                    },
                ]
            }
        }
    },
}])


NON_ANSWER_DATA = json.dumps([{
    "id": "001",
    "data": {
        "Name": "asset01.jpg",
        "Image": "https://asset.samasource.org",
        "Height": "720",
        "Width": "1280"
    },
    "answers": {},
}])


def test_setup(tmp_path):
    # Temporary file
    path = tmp_path / "sub"
    path.mkdir()
    file = path / "data.json"
    file.write_text(DATA)

    dataSetImporter = SAMADatasetImporter(str(file))
    dataSetImporter.setup()

    # Mock detection
    result = []
    detection = {'Vehicle': 'other_vehicle', 'bounding_box': [
        0.05234375, 0.6013888888888889, 0.14609375, -0.325], 'label': 'other_vehicle'}
    result.append(fo.Detection(**detection))
    annotations = [
        {'https://asset.samasource.org': {'detections': fo.Detections(detections=result)}}]

    # Expected values
    expected_filenames = ['https://asset.samasource.org']
    expected_annotations = _set_default_id(
        'https://asset.samasource.org', '62a77c9ee750224a5f4a422e', annotations[0])
    setup_annotation_default_id = _set_default_id(
        'https://asset.samasource.org', '62a77c9ee750224a5f4a422e', dataSetImporter.annotations[0])

    assert expected_annotations == setup_annotation_default_id
    assert expected_filenames == dataSetImporter.filenames


def test_setup_without_annotations(tmp_path):
    path = tmp_path / "sub"
    path.mkdir()
    file = path / "data.json"
    file.write_text(NON_ANSWER_DATA)

    dataSetImporter = SAMADatasetImporter(str(file))
    dataSetImporter.setup()

    expected_annotations = [{'https://asset.samasource.org': {}}]
    expected_filenames = ['https://asset.samasource.org']

    assert expected_annotations == dataSetImporter.annotations
    assert expected_filenames == dataSetImporter.filenames


def _set_default_id(key, id, annotation):
    # Set a default id so the assert can be done
    detection = annotation[key]
    detections_default_id = detection['detections'].get_attribute_value(
        'detections')
    detections_default_id[0].set_attribute_value('id', id)

    return detection
