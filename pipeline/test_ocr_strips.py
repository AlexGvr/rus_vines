"""Coordinate safety for the offline strip OCR experiment."""
import unittest

import numpy as np
from PIL import Image

from experiment_ocr_strips import transformed_strip, original_box, merge
from ocr import Word


class StripCoordinates(unittest.TestCase):
    def test_inverse_rotation_and_strip_offset(self):
        for size in ((375,500),(104,424),(1169,3726)):
            for angle in (-12,0,12):
                with self.subTest(size=size,angle=angle):
                    _, inverse, scale, y0 = transformed_strip(Image.new('RGB',size),.4,1.,angle)
                    forward=np.linalg.inv(np.vstack([inverse,[0,0,1]]))[:2]
                    points=np.array([[10,size[1]*.5],[size[0]-10,size[1]*.5],
                                     [size[0]-10,size[1]*.8],[10,size[1]*.8]])
                    local=(points-[0,y0])*scale
                    polygon=local@forward[:,:2].T+forward[:,2]
                    box=original_box(polygon,inverse,scale,y0,size)
                    expected=[10,size[1]*.5,size[0]-10,size[1]*.8]
                    self.assertLessEqual(np.max(np.abs(np.array(box)-expected)),1.01)

    def test_repeated_word_elsewhere_is_not_erased(self):
        old=Word('РОЗОВОЕ',.9,(10,20,100,40))
        same=Word('розовое',.95,(11,21,101,41))
        other=Word('РОЗОВОЕ',.99,(10,200,100,220))
        self.assertEqual(merge([old],[same,other]),[same,other])


if __name__=='__main__':
    unittest.main()
