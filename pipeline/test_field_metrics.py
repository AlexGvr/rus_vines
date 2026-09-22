import unittest
from field_metrics import summarize


class MetricTests(unittest.TestCase):
    def test_wrong_answer_counts_as_fp_and_fn_and_missing_ref_stays(self):
        def row(gold, correct, shown, findable):
            return dict(gold=gold, correct=correct, in_top3=correct, in_top5=correct,
                        probability=1 if shown else 0, findable=findable,
                        rank_retrieval100=0 if correct else -1,
                        rank_shortlist=0 if correct else -1,
                        rank_geometry=0 if correct else -1,
                        rank_text=0 if correct else -1)
        r = summarize([row('a', True, True, True), row('b', False, True, True),
                       row('c', False, False, False), row('', False, True, False)], .5)
        self.assertEqual(r['all_positives']['n'], 3)
        self.assertEqual(r['all_positives']['top1'], 1/3)
        self.assertEqual(r['with_rejection_all_queries']['1'],
                         dict(tp=1, fp=2, fn=2, precision=1/3, recall=1/3, f1=1/3))


if __name__ == '__main__':
    unittest.main()
