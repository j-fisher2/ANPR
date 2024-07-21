import unittest
from script import read_plate

class TestReadPlate(unittest.TestCase):
    
    def setUp(self):
        self.expected_results = {
            'image1.jpg': 'HR.26.BR.9044',
            'image2.jpg': 'COVID19',
            'image3.jpg': 'BJY982',
            'image4.jpg': 'H982 FKL'
        }
    
    def test_read_plate(self):
        for image_name, expected_plate_number in self.expected_results.items():
            result = read_plate(image_name)
            self.assertEqual(result, expected_plate_number, f'Failed for {image_name}')

unittest.main()