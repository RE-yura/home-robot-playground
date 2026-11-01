"""Summarize final-evaluation runs: k successes of n trials and the one-sided 95 % Clopper-Pearson lower bound.
A record with zero simulated time is invalid; everything else counts, including an RTC queue underflow and a run that
left no record (listed for the operator). Assisted rows also report the assist interventions; walls-criterion rows also report the rim successes. Usage: final_summary.py RUN_DIR... (each with record.json, named <label>-<seed>)."""
import json,math,sys
from pathlib import Path

def lower_bound(k,n,alpha=0.05):
    """p such that P(X >= k | n, p) = alpha (one-sided Clopper-Pearson); 0 when k = 0."""
    if k==0:return 0.0
    tail=lambda p:sum(math.comb(n,i)*p**i*(1-p)**(n-i)for i in range(k,n+1))
    lo,hi=0.0,1.0
    for _ in range(200):
        mid=(lo+hi)/2
        if tail(mid)<alpha:lo=mid
        else:hi=mid
    return lo

PROTOCOL_KEYS=('assist','delaySteps','executionSteps','guidance','pauseForGripperChanges','prefixConditioning','chunkOffsetSteps','lateHoldSteps')

def protocol(record):
    """What must be identical across one row: model, executor protocol and the continuous schedule (not the playback rate)."""
    schedule=record.get('rtcSchedule')or {}
    return json.dumps({'model':record.get('model'),'protocol':record.get('protocol'),'members':record.get('members',4),'noise':record.get('noise','gaussian'),
                       'successCriterion':record.get('successCriterion','rim'),**{k:schedule.get(k)for k in PROTOCOL_KEYS}},sort_keys=True)

def summarize(runs):
    """Invalid: a record whose run never reached physics (zero simulated time). A run without any record cannot show
    whether physics started, so it counts as a failure and is listed in noRecord for the operator to justify."""
    invalid,valid,missing=[],[],[]
    for run in runs:
        record=run.get('record')
        if record is None:missing.append({'seed':run['seed'],'reason':run.get('driverError')or'no record'});continue
        if not record['result']['elapsed']>0:invalid.append({'seed':run['seed'],'reason':record.get('error')or'no physics step'})
        else:valid.append((run['seed'],record))
    if len({protocol(r)for _,r in valid})>1:raise ValueError('One row must not mix protocols (model, executor, assist or schedule)')
    success=[(s,r)for s,r in valid if r['result']['status']=='success']
    audited=[(s,r)for s,r in success if r.get('quality')]
    trials=len(valid)+len(missing)
    out={'trials':trials,'successes':len(success),'invalid':invalid,'noRecord':missing,
         'failures':[{'seed':s,'status':r['result']['status'],'reason':r['result'].get('reason')or r.get('error'),'simulationS':round(r['result']['elapsed'],1)}for s,r in valid if r['result']['status']!='success']
                   +[{'seed':m['seed'],'status':'no-record','reason':m['reason'],'simulationS':None}for m in missing],
         'underflowFailures':sum('underflow'in(r.get('error')or'')for _,r in valid),
         'successAtMost100N':sum(r['quality']['peakUnintendedForce']<=100 for _,r in audited),'successesWithoutForceAudit':len(success)-len(audited),
         'lowerBound95':round(lower_bound(len(success),trials),4)if trials else None,
         'successCriterion':valid[0][1].get('successCriterion','rim')if valid else None}
    if out['successCriterion']=='walls':
        # The run ends at the walls success; rimSuccesses counts the ones whose last 2 s also met the original rim box.
        rim=sum(r['result'].get('rimSettledFor',0)>=2-1e-8 for _,r in success)
        out.update({'rimSuccesses':rim,'rimLowerBound95':round(lower_bound(rim,trials),4)if trials else None})
    if any((r.get('rtcSchedule')or {}).get('assist')for _,r in valid):
        events=[(s,r,len(r.get('assistEvents')or []))for s,r in valid]
        out.update({'assistInterventions':sum(n for *_,n in events),'runsWithAssist':sum(1 for *_,n in events if n),
                    'assistedSuccesses':sum(1 for _,r,n in events if n and r['result']['status']=='success'),
                    'assistEventsBySeed':{s:r.get('assistEvents')for s,r,n in events if n}})
    return out

if __name__=='__main__':
    runs=[]
    for folder in map(Path,sys.argv[1:]):
        path=folder/'record.json';runs.append({'seed':int(folder.name.rsplit('-',1)[1]),'record':json.loads(path.read_text())if path.exists()else None,
                                               'driverError':None if path.exists()else'no record'})
    print(json.dumps(summarize(runs),indent=1))
