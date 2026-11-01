"""Explicit weights-only recovery/control comparison; ordinary resume stays strict."""
import json
from hsr_training import sha
from colab_recording import verify_checkpoint

def verify_recovery_initializer(root,manifest_sha256,current,dataset_manifest,*,control):
 if sha(root/'FILES.json')!=manifest_sha256:raise ValueError('Recovery parent manifest hash mismatch')
 saved=json.loads((root/'hsr-checkpoint.json').read_text());verify_checkpoint(root,saved)
 keys=('robotModelId','chunkLength','executeSamples','sampleHz','policy','processor','lerobotRevision','policyInputs','trainingSettings')
 if any(saved.get(k)!=current.get(k)for k in keys):raise ValueError('Recovery parent policy/settings changed')
 if control:
  if any(saved.get(k)!=current.get(k)for k in ['datasetSha256','normalizationSha256']):raise ValueError('Control data changed')
 else:
  review=dataset_manifest.get('review',{})
  if dataset_manifest.get('datasetReady')is not True or review.get('status')!='complete'or review.get('unresolvedCritical',1)or review.get('unresolvedImportant',1):raise ValueError('Reviewed ready recovery data required')
  fork=dataset_manifest.get('recoveryFork',{})
  if (fork.get('schema')!='hsr-recovery-data-fork-v1'or fork.get('parentDatasetSha256')!=saved['datasetSha256']or
      fork.get('parentNormalizationSha256')!=saved['normalizationSha256']or fork.get('roundTripVerified')is not True):raise ValueError('Recovery data derivation mismatch')
  if current['datasetSha256']==saved['datasetSha256']:raise ValueError('Candidate data did not change')
 return {'schema':'hsr-recovery-initialization-v1','mode':'weights-only','control':control,'optimizerState':'fresh','manifestSha256':manifest_sha256,'parentTrainedUpdates':saved['trainedUpdates'],'parentSourceCommit':saved['sourceCommit'],'parentDatasetSha256':saved['datasetSha256'],'parentNormalizationSha256':saved['normalizationSha256']}
