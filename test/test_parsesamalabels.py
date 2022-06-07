from .context import (SAMADatasetImporter)
import json

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

# def test_from_answer_to_detection():
#     dataSetImporter = SAMADatasetImporter()
#     detection = dataSetImporter._parse_sama_labels(DATA)

#     expected_detection = [{'https://asset.samasource.org': {'detections': < Detections: {
#         'detections': BaseList([
#             < Detection: {
#                 'id': '629f7927c9df31634dde56ad',
#                 'attributes': BaseDict({}),
#                 'tags': BaseList([]),
#                 'label': 'other_vehicle',
#                 'bounding_box': BaseList([
#                     0.05234375,
#                     0.6013888888888889,
#                     0.14609375,
#                     -0.325,
#                 ]),
#                 'mask': None,
#                 'confidence': None,
#                 'index': None,
#                 'Vehicle': 'other_vehicle',
#             } > ,
#         ]),
#     } > }}]

#     assert detection == expected_detection


def test_answer_with_non_labels():
  dataSetImporter = SAMADatasetImporter()
  detection = dataSetImporter._parse_sama_labels(NON_ANSWER_DATA)
  expected_detection = [{'https://asset.samasource.org': {}}]

  assert detection == expected_detection
  