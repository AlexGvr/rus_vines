import unittest
import numpy as np
from experiment_ranker import features, fit_pairwise, fold_assignments, predict
from searchcore import Candidate, demote_contradicted


class RankerTests(unittest.TestCase):
    def test_cached_veto_matches_production_rule(self):
        candidates = [dict(slug=str(i), cv=1-i*.1, inliers=0, coverage=0,
                           hard_conflicts=int(i<2), text_conflicts=0, text_support=0)
                      for i in range(4)]
        r = dict(image='image', gold='2', findable=True, top1='2', candidates=candidates)
        weights = np.array([1,0,0,0,0,0,0,0])
        predicted = predict([r], weights, preserve_hard_veto=True)[0]['order']
        class Channel:
            def hard_conflicts(self, slug, words, min_conf):
                return ['conflict'] if int(slug)<2 else []
        production = [Candidate(c['slug'],c['cv']) for c in candidates]
        demote_contradicted(production, Channel(), [], .6)
        self.assertEqual(predicted, [c.slug for c in production])

    def test_folds_keep_connected_wines_and_series_together(self):
        rows = [dict(image='a', series='s1', slug='w1'),
                dict(image='b', series='s2', slug='w1'),
                dict(image='c', series='s2', slug='w2'),
                dict(image='d', series='s3', slug='w2')]
        self.assertEqual(len(set(fold_assignments(rows).values())), 1)

    def test_pairwise_fit_prefers_consistently_stronger_candidate(self):
        def c(slug, cv):
            return dict(slug=slug, cv=cv, inliers=0, coverage=0,
                        hard_conflicts=0, text_conflicts=0, text_support=0)
        record = dict(gold='right', findable=True, candidates=[c('wrong', .2), c('right', .9)])
        weights = fit_pairwise([record])
        scores = features(record['candidates']) @ weights
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(scores[1], scores[0])


if __name__ == '__main__':
    unittest.main()
