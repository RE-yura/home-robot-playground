"""Training-time action-prefix conditioning (arXiv 2512.05964) for the HSR SmolVLA: masks, guard, identity, loss."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import hsr_training as h
import prefix_conditioning as p

PARENT='f289e2a8b06ca26216ca932700df4633272d1988d6e1253893ed5c5b9740468d'
FORK=['--sampling-from','fixture','--initialize-manifest-sha256',PARENT,'--initial-sampling','--initial-sampling-weight','8',
      '--sampling-low-rate','--rtc-training-max-delay','10','--steps','2000','--save-every','500']
BASE=['trainer','--dataset','fixture','--output','fixture-output','--source-commit','a'*40]

class PrefixMaskTests(unittest.TestCase):
    def test_lengths_cover_zero_to_max_as_contiguous_prefixes(self):
        generator=torch.Generator().manual_seed(1);seen=set()
        for _ in range(500):
            mask=p.sample_prefix_mask(1,16,10,generator);length=int(mask.sum())
            self.assertEqual(mask.shape,(1,16));self.assertEqual(mask.dtype,torch.bool)
            self.assertTrue(mask[0,:length].all());self.assertFalse(mask[0,length:].any());seen.add(length)
        self.assertEqual(seen,set(range(11)))

    def test_seeded_and_bounded(self):
        draw=lambda:[p.sample_prefix_mask(1,16,10,torch.Generator().manual_seed(7)).sum().item()for _ in range(3)]
        self.assertEqual(draw(),draw())
        self.assertEqual(p.RTC_TRAINING_MAX_DELAY,10)
        for bad in [16,-1]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):p.sample_prefix_mask(1,16,bad,torch.Generator())
        self.assertFalse(p.sample_prefix_mask(1,16,0,torch.Generator()).any())

class PrefixNoiseTests(unittest.TestCase):
    def test_offset_is_constant_over_the_prefix_and_limited_to_prefix_goals_and_real_axes(self):
        x=torch.zeros(3,16,32);mask=torch.zeros(3,16,dtype=torch.bool);mask[0,:5]=True;mask[2,:8]=True
        out=p.perturb_prefix(x,mask,sigma=.03,real_dims=6,generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(out[:,:,6:],x[:,:,6:]),'padded axes untouched')
        self.assertTrue(torch.equal(out[~mask],x[~mask]),'postfix goals untouched')
        self.assertTrue(torch.equal(out[1],x[1]),'no prefix, no change')
        self.assertTrue(torch.allclose(out[0,:5,:6],out[0,:1,:6].expand(5,6)),'one offset per sample over its prefix')
        self.assertGreater(float(out[0,0,:6].abs().sum()),0)
        a=p.perturb_prefix(x,mask,sigma=.03,real_dims=6,generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a,out),'seeded')
        self.assertTrue(torch.equal(p.perturb_prefix(x,mask,sigma=0.,real_dims=6,generator=torch.Generator()),x))
    def test_prefix_noise_requires_the_delay_eight_mug_fork(self):
        argv=BASE+FORK+['--prefix-noise','0.03'];argv[argv.index('--rtc-training-max-delay')+1]='8'
        argv[argv.index('--steps')+1]='8000';argv[argv.index('--save-every')+1]='2000'
        class ReachedCuda(Exception):pass
        with patch('sys.argv',argv+['--mug-data-fork']),patch.object(h.torch.cuda,'is_available',side_effect=ReachedCuda):
            with self.assertRaises(ReachedCuda):h.main()
        for bad in [argv,BASE+FORK+['--prefix-noise','0.03'],argv+['--mug-data-fork','--prefix-noise-bad']]:
            with self.subTest(bad=bad),patch('sys.argv',bad),patch.object(h.torch.cuda,'is_available',side_effect=AssertionError('GPU must not be consulted')):
                with self.assertRaises((ValueError,SystemExit)):h.main()

class GuardTests(unittest.TestCase):
    def test_prefix_training_requires_its_exclusive_parent_fork(self):
        class ReachedCuda(Exception):pass
        with patch('sys.argv',BASE+FORK),patch.object(h.torch.cuda,'is_available',side_effect=ReachedCuda):
            with self.assertRaises(ReachedCuda):h.main()
        bad=[BASE+FORK+extra for extra in [['--mug-release-loss'],['--relative-arm-actions'],['--action-offset-steps','2'],['--grasp-sampling']]]
        for flag,value in [('--steps','1000'),('--save-every','250'),('--initialize-manifest-sha256','b'*64),('--rtc-training-max-delay','8')]:
            argv=BASE+FORK;argv[argv.index(flag)+1]=value;bad.append(argv)
        argv=BASE+FORK;argv.remove('--sampling-low-rate');bad.append(argv)
        for argv in bad:
            with self.subTest(argv=argv),patch('sys.argv',argv),patch.object(h.torch.cuda,'is_available',side_effect=AssertionError('GPU must not be consulted')):
                with self.assertRaises((ValueError,SystemExit)):h.main()

    def test_delay_eight_requires_its_own_8000_update_budget(self):
        class ReachedCuda(Exception):pass
        def argv(delay,steps,save,accumulate='4'):
            a=BASE+FORK+['--accumulate',accumulate];a[a.index('--rtc-training-max-delay')+1]=delay
            a[a.index('--steps')+1]=steps;a[a.index('--save-every')+1]=save;return a
        with patch('sys.argv',argv('8','8000','2000')),patch.object(h.torch.cuda,'is_available',side_effect=ReachedCuda):
            with self.assertRaises(ReachedCuda):h.main()
        for bad in [argv('8','2000','500'),argv('10','8000','2000'),argv('8','8000','500'),argv('8','8000','2000','2'),argv('9','8000','2000')]:
            with self.subTest(argv=bad),patch('sys.argv',bad),patch.object(h.torch.cuda,'is_available',side_effect=AssertionError('GPU must not be consulted')):
                with self.assertRaises((ValueError,SystemExit)):h.main()

    def identity(self,extra):
        class StopBeforeDownload(Exception):pass
        built=[];original=h.checkpoint_identity
        def capture(*args):
            built.append(original(*args));return built[-1]
        with tempfile.TemporaryDirectory()as tmp:
            parent=Path(tmp)/'parent';parent.mkdir()
            saved=original({'fixture':'synthetic'},{'fixture':'synthetic'},'a'*40,4)
            saved['trainingSettings'].update(sampling={'firstFrames':20,'earlyWeight':8,'laterWeight':1},learningRate=2e-5)
            saved.update(trainedUpdates=1000,samplingFork={'profile':'first20x8'})
            (parent/'hsr-checkpoint.json').write_text(json.dumps(saved));wire=b'[]';(parent/'FILES.json').write_bytes(wire)
            argv=['trainer','--dataset',str(Path(tmp)/'data'),'--output',str(Path(tmp)/'out'),'--source-commit','a'*40,
                  '--sampling-from',str(parent),'--initialize-manifest-sha256',hashlib.sha256(wire).hexdigest(),
                  '--initial-sampling','--initial-sampling-weight','8','--sampling-low-rate','--steps','2000','--save-every','500']+extra
            with patch('sys.argv',argv),patch.object(h.torch.cuda,'is_available',return_value=True),\
                 patch.object(h,'OFFSET_PARENT_MANIFEST',hashlib.sha256(wire).hexdigest()),patch.object(h,'verify_dataset',return_value=({'fixture':'synthetic'},{'fixture':'synthetic'})),\
                 patch.object(h,'checkpoint_identity',side_effect=capture),patch('colab_recording.verify_checkpoint'),\
                 patch('huggingface_hub.snapshot_download',side_effect=StopBeforeDownload):
                with self.assertRaises(StopBeforeDownload):h.main()
        return built[-1]

    def test_identity_records_the_maximum_delay_only_when_set(self):
        default=self.identity([]);conditioned=self.identity(['--rtc-training-max-delay','10'])
        self.assertNotIn('rtcTrainingMaxDelay',default['trainingSettings'])
        self.assertEqual(conditioned['trainingSettings']['rtcTrainingMaxDelay'],10)
        del conditioned['trainingSettings']['rtcTrainingMaxDelay'];self.assertEqual(conditioned,default)
        eight=self.identity(['--rtc-training-max-delay','8','--steps','8000','--save-every','2000'])
        self.assertEqual(eight['trainingSettings']['rtcTrainingMaxDelay'],8)

@unittest.skipUnless(os.environ.get('HSR_EXPORT_PARENT')and os.environ.get('HSR_OFFSET_DATASET'),'needs the public parent checkpoint and dataset')
class ParentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torch.utils.data import default_collate
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        parent,dataset=Path(os.environ['HSR_EXPORT_PARENT']),Path(os.environ['HSR_OFFSET_DATASET'])
        config=h.training_config(PreTrainedConfig.from_pretrained(parent/'policy'),parent/'processor')
        cls.policy=SmolVLAPolicy.from_pretrained(parent/'policy',config=config).float().eval()
        preprocess,_=make_smolvla_pre_post_processors(config,json.loads((dataset/'train'/'meta'/'stats.json').read_text()))
        validation=h.hsr_datasets(dataset,0)[1]
        cls.batch=preprocess(h.policy_batch(default_collate([validation[50]])))
        generator=torch.Generator().manual_seed(3)
        cls.noise=torch.randn((1,16,32),generator=generator).to(cls.batch['action'].device)
        cls.time=torch.tensor([.6]).to(cls.batch['action'].device)

    def test_zero_delay_matches_policy_forward(self):
        with torch.no_grad():
            reference,_=self.policy.forward(dict(self.batch),noise=self.noise.clone(),time=self.time.clone())
            loss,_=p.prefix_training_loss(self.policy,dict(self.batch),torch.zeros((1,16),dtype=torch.bool,device=self.noise.device),
                                          noise=self.noise.clone(),time=self.time.clone())
        self.assertLessEqual(abs(loss.item()-reference.item()),1e-6)

    def test_uniform_scalar_time_matches_embed_suffix(self):
        model=self.policy.model;x=self.noise.clone()
        with torch.no_grad():
            expected=model.embed_suffix(x,self.time)[0];actual=p.embed_suffix_tokens(model,x,self.time[:,None].expand(1,16))[0]
        self.assertTrue(torch.allclose(expected,actual,atol=0,rtol=0))

    def test_prefix_tokens_are_clean_and_unscored(self):
        mask=torch.zeros((1,16),dtype=torch.bool,device=self.noise.device);mask[:,:3]=True;seen={}
        original=p.embed_suffix_tokens
        def spy(model,noisy,token_time):
            seen.update(noisy=noisy.clone(),time=token_time.clone());return original(model,noisy,token_time)
        actions=self.policy.prepare_action(self.batch)
        with torch.no_grad(),patch.object(p,'embed_suffix_tokens',side_effect=spy):
            loss,_=p.prefix_training_loss(self.policy,dict(self.batch),mask,noise=self.noise.clone(),time=self.time.clone())
            losses=p.prefix_flow_losses(self.policy.model,*p.model_inputs(self.policy,dict(self.batch)),actions,self.noise.clone(),self.time.clone(),mask)
        self.assertTrue(torch.equal(seen['noisy'][:,:3],actions[:,:3]));self.assertTrue((seen['time'][:,:3]==0).all())
        self.assertTrue((seen['time'][:,3:]==.6).all())
        valid=~(self.batch['action_is_pad']|mask)
        expected=(losses[:,:,:6]*valid.unsqueeze(-1)).sum()/(valid.sum()*6)
        self.assertLessEqual(abs(loss.item()-expected.item()),1e-6)

    def test_batched_prefix_loss_matches_concatenated_valid_mean(self):
        # The recipe's batched forward must average over every valid goal of the batch, not average per-sample means.
        from torch.utils.data import default_collate
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        dataset=Path(os.environ['HSR_OFFSET_DATASET']);validation=h.hsr_datasets(dataset,0)[1]
        preprocess,_=make_smolvla_pre_post_processors(self.policy.config,json.loads((dataset/'train'/'meta'/'stats.json').read_text()))
        samples=[validation[50],validation[400]];device=self.noise.device
        pair=preprocess(h.policy_batch(default_collate(samples)))
        generator=torch.Generator().manual_seed(5);noise=torch.randn((2,16,32),generator=generator).to(device);time=torch.tensor([.6,.3]).to(device)
        mask=torch.zeros((2,16),dtype=torch.bool,device=device);mask[0,:2]=True;mask[1,:7]=True
        with torch.no_grad():
            joint,_=p.prefix_training_loss(self.policy,dict(pair),mask,noise=noise.clone(),time=time.clone())
            total,count=0.,0
            for i,sample in enumerate(samples):
                one=preprocess(h.policy_batch(default_collate([sample])))
                loss,_=p.prefix_training_loss(self.policy,dict(one),mask[i:i+1],noise=noise[i:i+1].clone(),time=time[i:i+1].clone())
                valid=int((~(one['action_is_pad'].to(device)|mask[i:i+1])).sum())*6;total+=loss.item()*valid;count+=valid
        self.assertLessEqual(abs(joint.item()-total/count),1e-5)

    def test_vision_training_reaches_the_vision_tower_but_not_the_language_model(self):
        names=h.freeze_policy(self.policy,train_vision=True)
        try:
            mask=torch.zeros((1,16),dtype=torch.bool,device=self.noise.device);mask[:,:3]=True
            loss,_=p.prefix_training_loss(self.policy,dict(self.batch),mask,noise=self.noise.clone(),time=self.time.clone());loss.backward()
            grads={n:q.grad for n,q in self.policy.named_parameters()}
            vision=[n for n in names if n.startswith(h.VISION_PREFIXES[0])]
            self.assertTrue(vision and any(grads[n]is not None and grads[n].abs().sum()>0 for n in vision))
            self.assertTrue(all(grads[n]is None for n in grads if n.startswith('model.vlm_with_expert.vlm.model.text_model.')))
        finally:
            for q in self.policy.parameters():q.requires_grad_(False);q.grad=None

    def sampling_inputs(self):
        return p.model_inputs(self.policy,dict(self.batch))

    def test_masked_embedding_equals_per_token_time(self):
        mask=torch.zeros((1,16),dtype=torch.bool,device=self.noise.device);mask[:,:5]=True
        token_time=torch.where(mask,torch.zeros_like(mask,dtype=torch.float32),self.time[:,None].expand(1,16))
        with torch.no_grad():
            expected=p.embed_suffix_tokens(self.policy.model,self.noise,token_time)[0]
            actual=p.embed_suffix_masked(self.policy.model,self.noise,self.time,mask)[0]
        self.assertTrue(torch.allclose(expected,actual,atol=0,rtol=0))

    def test_sampling_without_prefix_matches_native_sampling(self):
        mask=torch.zeros((1,16),dtype=torch.bool,device=self.noise.device)
        with torch.no_grad():
            expected=self.policy.model.sample_actions(*self.sampling_inputs(),noise=self.noise.clone())
            actual=p.sample_with_prefix(self.policy.model,*self.sampling_inputs(),self.noise.clone(),torch.zeros_like(self.noise),mask)
        self.assertLessEqual((expected-actual).abs().max().item(),1e-5)

    def test_sampling_keeps_the_prefix_and_matches_lerobot_hard_prefix(self):
        from lerobot.policies.common.flow_matching import euler_integrate
        model=self.policy.model;inputs=self.sampling_inputs();mask=torch.zeros((1,16),dtype=torch.bool,device=self.noise.device);mask[:,:3]=True
        prefix=self.policy.prepare_action(self.batch)
        with torch.no_grad():
            actual=p.sample_with_prefix(model,*inputs,self.noise.clone(),prefix,mask)
            cache=p.prefix_cache(model,*inputs)
            reference=euler_integrate(lambda x,t:p.denoise_tokens(model,cache,x,t),self.noise.clone(),model.config.num_steps,
                                      hard_prefix=prefix,hard_prefix_mask=mask[...,None].expand_as(prefix))
        self.assertTrue(torch.equal(actual[:,:3],prefix[:,:3]))
        self.assertLessEqual((reference-actual).abs().max().item(),1e-5)

if __name__=='__main__':unittest.main()
