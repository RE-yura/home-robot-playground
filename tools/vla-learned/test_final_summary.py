"""Final-evaluation accounting: only runs that never reached physics are invalid; underflow is a failure."""
import unittest
import final_summary as f

def run(seed,status='success',elapsed=22.0,error=None,force=40.0):
    return {'seed':seed,'record':{'result':{'status':status,'elapsed':elapsed},'error':error,'quality':{'peakUnintendedForce':force}if force is not None else None}}

class FinalSummary(unittest.TestCase):
    def test_underflow_is_a_failure_not_invalid(self):
        s=f.summarize([run(3030),run(3031,'running',17.1,'RTC queue underflow')])
        self.assertEqual((s['successes'],s['trials'],s['invalid'],s['underflowFailures']),(1,2,[],1))
    def test_only_records_without_physics_are_invalid(self):
        s=f.summarize([run(3030),run(3032,'running',0.0,'parity failed')])
        self.assertEqual(s['trials'],1);self.assertEqual([i['seed']for i in s['invalid']],[3032])
    def test_a_missing_record_is_a_listed_failure_not_invalid(self):
        # Without a record nobody can tell whether physics had started; it counts as a failure until justified.
        s=f.summarize([run(3030),{'seed':3031,'record':None,'driverError':'no record'}])
        self.assertEqual((s['trials'],s['successes'],s['invalid']),(2,1,[]));self.assertEqual([i['seed']for i in s['noRecord']],[3031])
    def test_mixed_protocols_are_refused(self):
        a,b=run(3030),run(3031)
        a['record']['rtcSchedule']={'assist':True,'delaySteps':5};b['record']['rtcSchedule']={'delaySteps':5}
        with self.assertRaisesRegex(ValueError,'protocol'):f.summarize([a,b])
    def test_late_reply_grace_is_part_of_the_protocol(self):
        a,b=run(3030),run(3031)
        a['record']['rtcSchedule']={'delaySteps':5,'lateHoldSteps':3};b['record']['rtcSchedule']={'delaySteps':5}
        with self.assertRaisesRegex(ValueError,'protocol'):f.summarize([a,b])
    def test_members_and_noise_are_part_of_the_protocol(self):
        a,b=run(3030),run(3031);a['record']['members']=1;a['record']['noise']='zero'
        with self.assertRaisesRegex(ValueError,'protocol'):f.summarize([a,b])
        c=run(3032);c['record']['members']=4;c['record']['noise']='gaussian';f.summarize([b,c])
    def test_assist_interventions_are_counted_per_run(self):
        a,b=run(3030),run(3031,'failed')
        for r in (a,b):r['record']['rtcSchedule']={'assist':True,'delaySteps':5}
        a['record']['assistEvents']=[{'time':20,'reason':'regrasp-loop'},{'time':40,'reason':'stall'}]
        s=f.summarize([a,b])
        self.assertEqual((s['assistInterventions'],s['runsWithAssist'],s['assistedSuccesses']),(2,1,1))
    def test_force_limited_success_counts_only_audited_runs_at_most_100_n(self):
        s=f.summarize([run(3030,force=99.9),run(3031,force=100.1),run(3032,force=None),run(3033,'failed',force=10)])
        self.assertEqual(s['successAtMost100N'],1);self.assertEqual(s['successesWithoutForceAudit'],1)
    def test_success_criterion_is_part_of_the_protocol(self):
        a,b=run(3030),run(3031);a['record']['successCriterion']='walls'
        with self.assertRaisesRegex(ValueError,'protocol'):f.summarize([a,b])
        b['record']['successCriterion']='rim';self.assertEqual(f.summarize([b,run(3032)])['successCriterion'],'rim')
    def test_walls_rows_also_report_successes_that_met_the_rim_criterion(self):
        rows=[run(3030),run(3031),run(3032,'failed')]
        for r,rim in zip(rows,(2.0,0.4,0.0)):r['record']['successCriterion']='walls';r['record']['result']['rimSettledFor']=rim
        s=f.summarize(rows)
        self.assertEqual((s['successCriterion'],s['successes'],s['rimSuccesses']),('walls',2,1))
        self.assertEqual(s['rimLowerBound95'],round(f.lower_bound(1,3),4))
        self.assertNotIn('rimSuccesses',f.summarize([run(3030)]))
    def test_clopper_pearson_lower_bound(self):
        self.assertAlmostEqual(f.lower_bound(60,60),0.9513,places=4)
        self.assertAlmostEqual(f.lower_bound(59,60),0.9234,places=4)
        self.assertEqual(f.lower_bound(0,10),0.0)

if __name__=='__main__':unittest.main()
