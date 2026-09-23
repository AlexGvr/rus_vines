import unittest
from experiment_paddle_groups import line_groups


class LineGroups(unittest.TestCase):
    def test_adjacent_fragments_merge_but_other_lines_do_not(self):
        boxes = [[10,10,80,30],[85,12,150,32],[10,60,150,80]]
        self.assertEqual(line_groups(boxes),[[0,1]])

    def test_distant_or_different_size_text_does_not_merge(self):
        self.assertEqual(line_groups([[0,0,30,10],[90,0,120,10],[0,0,30,50]]),[])


if __name__=='__main__':
    unittest.main()
