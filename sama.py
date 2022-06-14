
import math
from enum import Enum
from fileinput import filename

import eta.core.serial as etas
import fiftyone as fo
import fiftyone.core.metadata as fom
import fiftyone.types as fot
import fiftyone.utils.data as foud
import numpy as np
from charset_normalizer import detect
from PIL import Image


class Substring(Enum):
    HEIGHT = 'Height'
    WIDTH = 'Width'
    HTTPS = 'https'


class SearchIn(Enum):
    KEY = 'key'
    VALUE = 'value'


class Sama(Enum):
    DATA = 'data'
    LAYERS = 'layers'
    VECTOR_TAGGING = 'vector_tagging'
    SHAPES = 'shapes'
    TAGS = 'tags'
    POINTS = 'points'
    TYPE = 'type'
    ANSWERS = 'answers'
    VECTOR_GO_ANSWER = '_vector'
    URL = 'url'


class SAMADatasetImporter(foud.LabeledImageDatasetImporter):
    """ Import SAMA-formatted datasets into FiftyOne
        Args:
            dataset_dir (None): the dataset directory. This may be optional for
                some importers
            shuffle (False): whether to randomly shuffle the order in which the
                samples are imported
            seed (None): a random seed to use when shuffling
            max_samples (None): a maximum number of samples to import. By default,
                all samples are imported
            **kwargs: additional keyword arguments for your importer
        """

    def __init__(
        self,
        dataset_dir=None,
        shuffle=False,
        seed=None,
        max_samples=None,
        **kwargs,  # Add any other arguments you want
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

    def setup(self):
        annotations = self._parse_sama_labels(self.dataset_dir)
        self._filenames = [key for dict in annotations for key in dict.keys()]
        self._annotations = annotations

    @property
    def annotations(self):
        return self._annotations

    @property
    def filenames(self):
        return self._filenames

    def __eq__(self, other):
        return \
            self.annotations == other.annotations, \
            self.filenames == other.filenames

    def __len__(self):
        return len(self._filenames)  # Parsed in setup()

    def __iter__(self):
        self._iter_filenames = iter(self._filenames)
        return self

    def __next__(self):
        filename = next(self._iter_filenames)
        sample_labels = self._get_sample_labels(filename)
        metadata = fom.ImageMetadata.build_for(filename)

        return filename, metadata, sample_labels

    def _get_sample_labels(self, filename):
        for element in self._annotations:
            if filename in element:
                return element[filename]

    @property
    def label_cls(self):
        return None

    @property
    def has_image_metadata(self):
        return True

    @property
    def has_dataset_info(self):
        return False

    def _parse_sama_labels(self, dataset_dir):
        """This method returns a list with annotations in voxel51 format

        Each annotation is a dictionary. The key corresponds to the asset
        s3 URI and the value contains the scene attributes and a list of
        detections.
        """
        labels_dict = etas.load_json(dataset_dir)
        result = []
        detections = []
        for element in labels_dict:
            layer = self._get_answers_layers(element)
            url = self._get_url(element)

            if layer is not None:
                detections = self._from_answer_to_detection(layer, element)
                scene_attributes = self._get_answer_scene_attributes(element)
                result.append({url: {**detections, **scene_attributes}})
            else:
                result.append({url: {}})

        return result

    def _from_answer_to_detection(self, layers, element):
        """This method returns a valid detection dictionary

        Voxel51 provides fo.Detections class that converts
        the input data into a detection
        Reference:
        https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#object-detection
        """
        result = []
        vectors = layers[Sama.VECTOR_TAGGING.value]
        for vector in vectors:
            shapes = vector[Sama.SHAPES.value]
            for shape in shapes:
                tags = [{x: y} for x, y in shape['tags'].items()]
                points = self._from_points_to_voxel51_bounding_box(
                    shape[Sama.POINTS.value], element)
                label = list(shape[Sama.TAGS.value].values())[0]
                type = shape[Sama.TYPE.value]
                tags.append({"label": label})
                tags.append({"bounding_box": points})
                tags.append({"type": type})
                detection = {**shape[Sama.TAGS.value], **
                             {"bounding_box": points}, **{'label': label}}
                result.append(fo.Detection(**detection))

        return {'detections': fo.Detections(detections=result)}

    def _get_answer_scene_attributes(self, element):
        """This method returns the scene attributes in an answer

        The scene attributes describe qualities of the asset. It can be
        daytime, a place, asset quality etc.
        """
        result = {}
        for key, value in element[Sama.ANSWERS.value].items():
            if not isinstance(value, dict):
                result[key] = value
        return result

    def _get_answers_layers(self, element):
        """This method returns the layers of an annotation

        Layers is a nested input inside :answers" the parent key of layers can
        change. If the project was set up on Sama Go the key corresponds to
        _vector on Sama Platform the key is set up by the user.

        {"answers":{
            "<key>":{
                "layers":
                {}
            }
        }
        """
        # Is a Sama Go project
        if "_vector" in element[Sama.ANSWERS.value]:
            if element[Sama.ANSWERS.value] != {}:
                return element[Sama.ANSWERS.value][Sama.VECTOR_GO_ANSWER.value]

        # Is a platform project
        else:
            # check answer is not empty
            if element[Sama.ANSWERS.value] != {}:
                answers = element[Sama.ANSWERS.value].values()
                return self._search_platform_layers(answers)

    def _search_platform_layers(self, answers):
        """This method returns a dictionary with the answer

        Only one value inside answers is a dictionary
        and this corresponds to layers.
        This is a way to get the value without knowing the key name
        """
        for answer in answers:
            if isinstance(answer, dict) and len(answer) != 0:
                return answer.get(Sama.LAYERS.value)

    def _get_url(self, element):
        return self._search_substring_in_dictionary(
            element[Sama.DATA.value], Substring.HTTPS.value, SearchIn.VALUE.value)

    def _from_points_to_voxel51_bounding_box(self, points, element):
        """This method returns a list with the values needed to draw a
        bounding box

        Sama annotation points follow the structure:[[x1,y1],[x1,y2],
                                                     [x2,y2],[x2,y1]]
        Voxel51 requires the format:[x1, y1, width , height]
        Reference:
        https://voxel51.com/docs/fiftyone/recipes/adding_detections.html?highlight=bounding%20box
        """

        # Get key values based on suffix "Width" and "Height"
        image_height = self._search_substring_in_dictionary(
            element[Sama.DATA.value], Substring.HEIGHT.value, SearchIn.KEY.value)
        image_width = self._search_substring_in_dictionary(
            element[Sama.DATA.value], Substring.WIDTH.value, SearchIn.KEY.value)

        if image_height is None or image_width is None:
            image_width, image_height = self._calculate_image_dimensions(
                element)

        rectangle = RectanglePoints(np.array(points))

        top_left_x_point = rectangle.get_top_left_point().get_x
        top_left_y_point = rectangle.get_top_left_point().get_y
        bottom_right_x_point = rectangle.get_bottom_right_point().get_x
        bottom_right_y_point = rectangle.get_bottom_right_point().get_y

        relative_width = (bottom_right_x_point -
                          top_left_x_point) / int(image_width)
        relative_height = (bottom_right_y_point -
                           top_left_y_point) / int(image_height)

        return [top_left_x_point / int(image_width), top_left_y_point /
                int(image_height), relative_width, relative_height]

    def _search_substring_in_dictionary(
            self, dictionary, substr, dictionary_item):
        """This method returns a string that contains a specific substring.

        dictionary_item specifies where to search the substring in a key or in a value
        """
        for key, value in dictionary.items():
            value_conditional = dictionary_item == 'value' and substr in value
            key_conditional = dictionary_item == 'key' and substr in key
            if value_conditional or key_conditional:
                return value

    def _calculate_image_dimensions(self, element):
        image = Image.open(element[Sama.DATA.value][Sama.URL.value])
        return image.size


class CustomLabeledImageDataset(fot.LabeledImageDataset):

    """Custom labeled image dataset type."""

    def get_dataset_importer_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.importers.LabeledImageDatasetImporter`
        class for importing datasets of this type from disk.

        Returns:
            a:class:`fiftyone.utils.data.importers.LabeledImageDatasetImporter`
            class
        """

        return SAMADatasetImporter

    def get_dataset_exporter_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.exporters.LabeledImageDatasetExporter`
        class for exporting datasets of this type to disk.

        Returns:
            a:class:`fiftyone.utils.data.exporters.LabeledImageDatasetExporter`
            class
        """
        # Return your custom LabeledImageDatasetExporter class here
        pass


class Point():
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __eq__(self, other):
        return self._x == other.get_x and self._y == other.get_y

    @property
    def get_x(self):
        return self._x

    @property
    def get_y(self):
        return self._y


class Points(object):
    """Wrapper around np arrays that represent annotation points.
    """

    def __init__(self, coordinates):
        self._coordinates = coordinates

    def __eq__(self, other):
        return self._coordinates.tolist() == other._coordinates.tolist()

    @property
    def get_coordinates(self):
        return self._coordinates


class VectorPoints(Points):
    """Abstract class for vector shapes
    It calculates the max and min points over axis
    take() is a numpy function for get an specific index from a np.array
    """

    def __init__(self, coordinates):
        super().__init__(coordinates)
        if len(self.get_coordinates) == 0:
            raise SAMADatasetImporterException(
                'ERROR, the shape has empty coordinates')
        all_x = [np.take(x, 0) for x in self.get_coordinates]
        all_y = [np.take(x, 1) for x in self.get_coordinates]
        self._min_x = min(all_x)
        self._max_x = max(all_x)
        self._min_y = min(all_y)
        self._max_y = max(all_y)

    @property
    def max_x(self):
        return self._max_x

    @property
    def min_x(self):
        return self._min_x

    @property
    def max_y(self):
        return self._max_y

    @property
    def min_y(self):
        return self._min_y


class RectanglePoints(VectorPoints):
    """This class have only the points of a rectangle"""

    def __init__(self, coordinates):
        super().__init__(coordinates)
        if len(self.get_coordinates) != 4:
            raise SAMADatasetImporterException(
                'ERROR, the rectangle points are not valid')

    def calculate_width_and_height(self):
        """This method returns a tuple with the width and height of a rectangle.
        It takes the min(x,y) point as reference and the corresponding points
        to create a 90 degrees angle, this combination always generate width and height.
        """
        reference_coordinate = [self.min_x, self.min_y]
        coordinate_for_width_line = [self.max_x, self.min_y]
        coordinate_for_height_line = [self.min_x, self.max_y]
        height = self._find_distance(
            reference_coordinate, coordinate_for_width_line)
        width = self._find_distance(
            reference_coordinate, coordinate_for_height_line)

        return width, height

    def _find_distance(self, points_1, points_2):
        distance = math.sqrt(
            (points_2[0] - points_1[0]) ** 2 + (points_2[1] - points_1[1])**2)
        return int(distance)

    def get_top_left_point(self):
        return Point(self.min_x, self.max_y)

    def get_bottom_right_point(self):
        return Point(self.max_x, self.min_y)


class SAMADatasetImporterException(Exception):
    def __init__(self, detailed_error):
        self._detailed_error = detailed_error

    def __str__(self):
        return 'details: {0}'.format(
            self._detailed_error)
