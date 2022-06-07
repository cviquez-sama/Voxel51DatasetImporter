import os
import sys

from sama import(
    Points,
    Point,
    VectorPoints,
    RectanglePoints,
    SAMADatasetImporter,
    SAMADatasetImporterException,
    Substring,
    SearchIn)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
