import copy
import json
import unittest
from bound_metrics import SCHEMA, align, digest, roster_digest, measure_bound


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(recording=n, pcm_sha256=str(i)*64, samples=16000,
                          frames=40, labels=y) for n,i,y in [('a',1,[1,0]),('b',2,[0,1])]]
        self.doc = dict(schema=SCHEMA, results=[dict(recording=r['recording'], pcm_sha256=r['pcm_sha256'], events=[]) for r in self.rows])
        self.doc['results'][0]['events']=[dict(keyword_id=1, decision_row=10, score=.8, recording_id='a')]

    def test_valid_reordered_results_align_by_id(self):
        original=align(self.rows,self.doc)
        self.doc['results'].reverse()
        self.assertEqual(original,align(self.rows,self.doc))

    def test_row_order_changes_no_identity_digest(self):
        self.assertEqual(roster_digest(self.rows),roster_digest(self.rows[::-1]))

    def test_missing_result_rejected(self):
        self.doc['results'].pop()
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_extra_result_rejected(self):
        self.doc['results'].append(dict(recording='c',pcm_sha256='3'*64,events=[]))
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_duplicate_result_rejected(self):
        self.doc['results'].append(copy.deepcopy(self.doc['results'][0]))
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_duplicate_input_rejected(self):
        with self.assertRaises(ValueError):align(self.rows+[self.rows[0]],self.doc)

    def test_pcm_swap_rejected(self):
        self.doc['results'][0]['pcm_sha256']='2'*64
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_missing_pcm_rejected(self):
        del self.rows[0]['pcm_sha256']
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_unbound_old_list_rejected(self):
        with self.assertRaises(ValueError):align(self.rows,[[],[]])

    def test_unknown_schema_rejected(self):
        self.doc['schema']='v0'
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_nested_event_mismatch_rejected(self):
        self.doc['results'][0]['events'][0]['recording_id']='b'
        with self.assertRaises(ValueError):align(self.rows,self.doc)

    def test_malformed_events_rejected(self):
        for value in [None,{},[dict(keyword_id=True,decision_row=1,score=.5)],
                      [dict(keyword_id=1,decision_row=-1,score=.5)],
                      [dict(keyword_id=1,decision_row=1,score=float('nan'))]]:
            with self.subTest(value=value):
                d=copy.deepcopy(self.doc);d['results'][0]['events']=value
                with self.assertRaises(ValueError):align(self.rows,d)

    def test_score_domain_endpoints_and_outside(self):
        for value in (-.01, 1.000001):
            self.doc['results'][0]['events'][0]['score']=value
            with self.assertRaises(ValueError):align(self.rows,self.doc)
        for value in (0, 1):
            self.doc['results'][0]['events'][0]['score']=value
            self.assertEqual(align(self.rows,self.doc)[0][0]['score'],value)

    def test_metric_called_only_after_validation(self):
        called=[]
        def metric(rows,events,model,window):
            called.append(1)
            return [r['recording'] for r in rows],events,model,window
        self.doc['results'].reverse()
        result=measure_bound(self.rows,self.doc,metric=metric,window='pre_eof')
        self.assertEqual(result[0],['a','b']);self.assertEqual(result[1][0][0]['recording_id'],'a')
        self.doc['results'].pop()
        with self.assertRaises(ValueError):measure_bound(self.rows,self.doc,metric=metric,window='pre_eof')
        self.assertEqual(len(called),1)

    def test_frame_and_label_boundaries(self):
        for key,value in [('frames',0),('samples',True),('labels',[True,0]),('labels',[1])]:
            with self.subTest(key=key):
                rows=copy.deepcopy(self.rows);rows[0][key]=value
                with self.assertRaises(ValueError):measure_bound(rows,self.doc,metric=lambda *a:None,window='pre_eof')
        self.doc['results'][0]['events'][0]['decision_row']=40
        with self.assertRaises(ValueError):measure_bound(self.rows,self.doc,metric=lambda *a:None,window='pre_eof')

    def test_inputs_not_mutated(self):
        before=copy.deepcopy(self.doc);out=align(self.rows,self.doc);out[0][0]['score']=0
        self.assertEqual(before,self.doc)

if __name__=='__main__':unittest.main()
