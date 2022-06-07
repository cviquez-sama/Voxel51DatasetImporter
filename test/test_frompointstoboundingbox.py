from .context import (SAMADatasetImporter)


DATA = {
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
                  "tags": { "Vehicle": "other_vehicle" },
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
  }

def test_get_voxel51_bouding_box():
    dataSetImporter = SAMADatasetImporter()
    bounding_box = dataSetImporter._from_points_to_voxel51_bounding_box([
                    [67, 199],
                    [254, 199],
                    [67, 433],
                    [254, 433]
                  ], DATA)

    expected_bounding_box = [0.05234375, 0.6013888888888889, 0.14609375, -0.325]

    assert bounding_box == expected_bounding_box