
import fiftyone.utils.data as foud
import fiftyone.core.metadata as fom
import eta.core.serial as etas # Package comes with FiftyOne
import numpy as np
import math
from enum import Enum
import fiftyone.types as fot
import fiftyone as fo

class Substring(Enum):
    HEIGHT = 'Height'
    WIDTH = 'Width'
    HTTPS = 'https'


class SearchIn(Enum):
    KEY = 'key'
    VALUE = 'value'
    
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
        **kwargs, # Add any other arguments you want
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
        
    def __len__(self):
        return len(self._filenames) # Parsed in setup()
    
    # A convenient way to iterate through samples one at a time
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
            for key,value in element.items():
                if key == filename:
                    return value
        

    @property
    def label_cls(self):
        return None
        
    @property
    def has_image_metadata(self):
        return True # If you want to parse the ImageMetadata
        
    @property
    def has_dataset_info(self):
        return False # Unless you want to store any dataset-level information


    def _parse_sama_labels(self,dataset_dir):
        """This method returns a list with annotations in voxel51 format

        Each annotation is a dictionary. The key corresponds to the asset s3 URI
        ans the value contains the scene attributes and a list of detections.
        """
        labels_dict = etas.load_json(dataset_dir)
        result = []
        detections = []
        for element in labels_dict:
            layer = self._get_answers_layers(element)
            url = self._get_url(element)

            if layer != None :
                detections = self._from_answer_to_detection(layer, element)
                scene_attributes = self._get_answer_scene_attributes(element)
                result.append({url:{**detections, **scene_attributes}})
            else:
                result.append({url:{}})

        return result

    def _from_answer_to_detection(self,layers, element):
        """This method returns a valid detection dictionary

        Voxel51 provides fo.Detections class that converts 
        the input data into a detection 
        Reference: https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#object-detection
        """
        result = []
        vectors = layers['layers']['vector_tagging']
        for vector in vectors:
            shapes = vector['shapes']
            for shape in shapes:
                tags = [{x:y} for x,y in shape['tags'].items()]
                points = self._from_points_to_voxel51_bounding_box(shape['points'], element) 
                label = list(shape['tags'].values())[0]
                type = shape['type']
                tags.append({"label": label})
                tags.append({"bounding_box": points})
                tags.append({"type": type})

                aux = {**shape['tags'], **{"bounding_box": points}, **{'label':label}}
                result.append(fo.Detection(**aux))

        return {'detections': fo.Detections(detections=result)}



    def _get_answer_scene_attributes(self, element):
        """This method returns the scene attributes in an answer 

        The scene attributes describe qualities of the asset. It can be daytime,
        a place, asset quality etc.
        """
        result = {}
        for key,value in element['answers'].items():
            if not isinstance(value, dict):
                result[key]=value
        return result
        
    def _get_answers_layers(self, element):
        """This method returns the layers of an annotation

        Layers is a nested input inside :answers" the parent key 
        of layers can change. If the project was set up on 
        Sama Go the key corresponds to _vector on Sama Platform the 
        key is set up by the user.

        {"answers":{
            "<key>":{
                "layers":
                {}
            }
        }
        """
        # Is a Sama Go project
        if "_vector" in element['answers']:
            if element['answers'] != {}:
                return element['answers']['_vector']
            else:
                return None

        # Is a platform project
        else:
            print("ENTRA", element['answers'])
            # check answer is not empty
            if element['answers'] != {}:
                answers = element['answers'].values()
                return self._search_platform_layers(answers)           
            else:
                print('ENTRA2')
                return None
    
    
    def _search_platform_layers(self, answers):
        """This method returns a dictionary with the answer

        Only one value inside answers is a dictionary
        and this corresponds to layers. 
        This is a way to get the value without knowing the key name
        """
        for answer in answers:
            if isinstance(answer, dict) and len(answer) != 0:
                if "layers" in answer:
                    return answer
                
         

    def _get_url(self, element):
        return self._search_substring_in_dictionary(element['data'], Substring.HTTPS.value, SearchIn.VALUE.value)
                 
              
    def _from_points_to_voxel51_bounding_box(self, points, element):
        """This method returns a list with the values needed to draw a bounding box

        Sama annotation points follow the structure [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
        Voxel51 requires the format:[x1, y1, width , height]
        Reference: https://voxel51.com/docs/fiftyone/recipes/adding_detections.html?highlight=bounding%20box
        """
       
        # Get key values based on suffix "Width" and "Height"
        image_height = int(self._search_substring_in_dictionary(element['data'], Substring.HEIGHT.value, SearchIn.KEY.value))
        image_width = int(self._search_substring_in_dictionary(element['data'], Substring.WIDTH.value, SearchIn.KEY.value))

        rectangle = RectanglePoints(np.array(points))

        top_left_x_point = rectangle.get_top_left_point().get_x()
        top_left_y_point = rectangle.get_top_left_point().get_y()
        bottom_right_x_point = rectangle.get_bottom_right_point().get_x()
        bottom_right_y_point = rectangle.get_bottom_right_point().get_y()

        relative_width = (bottom_right_x_point - top_left_x_point) / image_width
        relative_height = (bottom_right_y_point - top_left_y_point) / image_height

        return [ top_left_x_point/image_width, top_left_y_point/image_height, relative_width , relative_height ]

        
    def _search_substring_in_dictionary(self, dictionary, substr, dictionary_item):
        """This method returns a string that contains a specific substring

        dictionary_item specifies where to search the substring in a key or in a value
        """
        for  key, value in dictionary.items():
            if dictionary_item == 'value':
                if substr in value:
                    return value
            if dictionary_item == 'key':
                if substr in key:
                    return value
  
        raise SAMADatasetImporterException(
            f'ERROR, Task meta data does not contains the value')
                 
           
              

class CustomLabeledImageDataset(fot.LabeledImageDataset):
    
    """Custom labeled image dataset type."""

    def get_dataset_importer_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.importers.LabeledImageDatasetImporter`
        class for importing datasets of this type from disk.

        Returns:
            a :class:`fiftyone.utils.data.importers.LabeledImageDatasetImporter`
            class
        """

        return SAMADatasetImporter


    def get_dataset_exporter_cls(self):
        """Returns the
        :class:`fiftyone.utils.data.exporters.LabeledImageDatasetExporter`
        class for exporting datasets of this type to disk.

        Returns:
            a :class:`fiftyone.utils.data.exporters.LabeledImageDatasetExporter`
            class
        """
        # Return your custom LabeledImageDatasetExporter class here
        pass

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

class Points(object):
    """Wrapper around np arrays that represent annotation points.
    """

    def __init__(self, coordinates):
        self._coordinates = coordinates

    def __eq__(self, other):
        return self._coordinates.tolist() == other._coordinates.tolist()

    def get_coordinates(self):
        return self._coordinates

class VectorPoints(Points):
    """Abstract class for vector shapes
    It calculates the max and min points over axis
    take() is a numpy function for get an specific index from a np.array
    """

    def __init__(self, coordinates):
        super().__init__(coordinates)
        if len(self.get_coordinates()) == 0:
            raise SAMADatasetImporterException(
                f'ERROR, the shape has empty coordinates')
        all_x = [np.take(x, 0) for x in self.get_coordinates()]
        all_y = [np.take(x, 1) for x in self.get_coordinates()]
        self._min_x = min(all_x)
        self._max_x = max(all_x)
        self._min_y = min(all_y)
        self._max_y = max(all_y)

    def get_max_x(self):
        return self._max_x

    def get_min_x(self):
        return self._min_x

    def get_max_y(self):
        return self._max_y

    def get_min_y(self):
        return self._min_y


class RectanglePoints(VectorPoints):
    """This class have only the points of a rectangle"""

    def __init__(self, coordinates):
        super().__init__(coordinates)
        if len(self.get_coordinates()) != 4:
            raise SAMADatasetImporterException(
                f'ERROR, the rectangle points are not valid')

    def calculate_width_and_height(self):
        """This method returns a tuple with the width and height of a rectangle.
        It takes the min(x,y) point as reference and the corresponding points
        to create a 90 degrees angle, this combination always generate width and height.
        """
        reference_coordinate = [self.get_min_x(), self.get_min_y()]
        coordinate_for_width_line = [self.get_max_x(), self.get_min_y()]
        coordinate_for_height_line = [self.get_min_x(), self.get_max_y()]
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
        return Point(self.get_min_x(), self.get_max_y())

    def get_bottom_right_point(self):
        return Point(self.get_max_x(), self.get_min_y())


class SAMADatasetImporterException(Exception):
    def __init__(self, detailed_error):
        self._detailed_error = detailed_error

    def __str__(self):
        return 'details: {0}'.format(
            self._detailed_error)


def main():
    dataset_type = CustomLabeledImageDataset
    dataset_dir = '/Users/cviquez/Downloads/2022-06-07_15-54_56-790_BUG_9801_delivery.json'

    # Import dataset
    dataset = fo.Dataset.from_dir(dataset_dir=dataset_dir, dataset_type=dataset_type)

    # Start session
    session = fo.launch_app(dataset)
    session.wait()
