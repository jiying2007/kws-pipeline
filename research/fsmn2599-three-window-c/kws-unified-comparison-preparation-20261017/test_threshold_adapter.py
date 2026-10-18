import itertools,sys,unittest
from threshold_before_refractory import ThresholdBeforeRefractory,ThreeWindowEvents,DecoderError
from test_posterior_events import row
from posterior_events import KEYWORDS
from test_three_window_events import candidate

class ThresholdFixtures(unittest.TestCase):
    def test_01_theta_zero_every_row_events_and_state_equal(self):
        tokens=[1401]+[0]*80+list(KEYWORDS[0][1])+[0]*45+list(KEYWORDS[1][1])
        a=ThreeWindowEvents('synthetic');b=ThresholdBeforeRefractory('synthetic',0)
        for t in tokens:
            self.assertEqual(a.step(row(t)),b.step(row(t)))
            self.assertEqual(a.summary(),b.summary())
            self.assertEqual(a._seen,b._seen)
            self.assertEqual([(w.start,w.cur_hyps) for w in a.windows],[(w.start,w.cur_hyps) for w in b.windows])
        self.assertEqual(a.finish(),b.finish());self.assertEqual(b.threshold_rejections,0)

    def test_02_low_then_high_within_refractory_is_not_postfilter_equivalent(self):
        rows=[row(t,.65) for t in KEYWORDS[0][1]]+[row(0)]*6+[row(t,.95) for t in KEYWORDS[1][1]]
        original=ThreeWindowEvents('synthetic');new=ThresholdBeforeRefractory('synthetic',.7)
        old_events=list(original.process(rows));new_events=list(new.process(rows))
        self.assertEqual([e['keyword_id'] for e in old_events],[1])
        self.assertEqual([e for e in old_events if e['score']>=.7],[])
        self.assertEqual([e['keyword_id'] for e in new_events],[2])
        self.assertAlmostEqual(new_events[0]['score'],.95**2)
        self.assertEqual(new.threshold_rejections,1)

    def test_03_chunk_boundaries_do_not_change_threshold_state(self):
        rows=[row(0)]*40+[row(t) for t in KEYWORDS[0][1]]+[row(0)]*18+[row(t) for t in KEYWORDS[1][1]]
        expected=None
        for chunk in (1,7,42,126):
            d=ThresholdBeforeRefractory('synthetic',.5);events=[]
            for start in range(0,len(rows),chunk):events.extend(d.process(rows[start:start+chunk]))
            actual=(events,d.finish(),d.threshold_rejections)
            if expected is None:expected=actual
            self.assertEqual(actual,expected)

    def test_04_score_equal_threshold_is_accepted_and_duplicates_remain_seen(self):
        d=ThresholdBeforeRefractory('synthetic',.7)
        low=candidate(1,(0,3,6,9),.6,decision=3)
        self.assertEqual(d._merge([low],3),[])
        high=dict(low,score=.9,decision=4)
        self.assertEqual(d._merge([high],4),[])
        self.assertEqual(d.duplicate_candidates,1)
        accepted=candidate(2,(30,33,36,39),.7,decision=13)
        self.assertEqual(len(d._merge([accepted],13)),1)

    def test_05_same_frame_order_and_multiple_events_are_deterministic(self):
        a=candidate(1,(0,3,6,9),.9);b=candidate(2,(60,63,66,69),.8)
        outputs=[]
        for sequence in itertools.permutations((a,b,dict(a))):
            d=ThresholdBeforeRefractory('synthetic',.7);outputs.append(d._merge(sequence,100))
        self.assertTrue(all(x==outputs[0] for x in outputs))
        self.assertEqual(len(outputs[0]),2)

    def test_06_eof_reset_and_invalid_threshold(self):
        for bad in (-.1,1.1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):ThresholdBeforeRefractory('synthetic',bad)
        d=ThresholdBeforeRefractory('synthetic',.9)
        list(d.process(row(t) for t in KEYWORDS[0][1]));d.finish()
        with self.assertRaises(DecoderError):d.step(row(0))
        d.reset_all('another');self.assertEqual(d.threshold_rejections,0)
        self.assertEqual(d.rows_decoded,0);self.assertEqual(d.last_accepted_end,-1)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
