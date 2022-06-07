import pytest   


from .context import (SAMADatasetImporter,
                      SAMADatasetImporterException,
                      Substring,
                      SearchIn)

DICTIONARY = {
                "id": "628fc10a2e74a204f90ab064",
                "project_id": "0001",
                "data": {
                    "Name": "0051.jpg",
                    "Image": "https://assets.com",
                    "Annotation Height": 720,
                    "Annotation Width": 1280
                }
            }

def test_search_substring_in_dictionary_value():
    dataSetImporter = SAMADatasetImporter()
    url = dataSetImporter._search_substring_in_dictionary(DICTIONARY['data'], Substring.HTTPS.value, SearchIn.VALUE.value)
    expected_url = "https://assets.com"

    assert url == expected_url
    

def test_search_substring_in_dictionary_key():
    dataSetImporter = SAMADatasetImporter()
    key = dataSetImporter._search_substring_in_dictionary(DICTIONARY['data'], Substring.HEIGHT.value, SearchIn.KEY.value)
    expected_key = "Annotation Height"

    assert key == expected_key


def test_non_valid_substring_in_dictionary():
    with pytest.raises(SAMADatasetImporterException) as exception:
        dataSetImporter = SAMADatasetImporter()
        dataSetImporter._search_substring_in_dictionary(DICTIONARY['data'], 'non-existing-susbtring', SearchIn.KEY)
        
    assert 'details: ERROR, Task meta data does not contains the value' == str(
        exception.value)