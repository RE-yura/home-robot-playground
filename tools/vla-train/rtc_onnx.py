"""Opt-in guided RTC inside a fixed ten-step ONNX Euler sampler.

Matches the installed LeRobot RTCProcessor's stop-gradient/identity correction.
This is explicitly not a full denoiser-Jacobian export. No inference, training,
weight loading or publication is performed by this metadata transformation.
"""
import copy
import math
import numpy as np
from onnx import helper as h, numpy_helper as nh, TensorProto as T


BASE_INPUTS={'head_rgb','hand_rgb','language_tokens','language_mask','hsr_state','initial_noise'}


def prefix_weights(delay, horizon, total=16):
    if any(type(x)is not int for x in (delay,horizon,total)) or not 0<=delay<=horizon<=total or total!=16:
        raise ValueError('Require integer0<=delay<=horizon<=16')
    middle=np.linspace(1,0,horizon-delay+2,dtype=np.float32)[1:-1]
    middle=middle*np.expm1(middle)/np.float32(math.e-1)
    return np.concatenate([np.ones(delay,np.float32),middle,np.zeros(total-horizon,np.float32)])


def add_rtc_guidance(model,max_gain=10.0):
    """Return a fresh graph; original tensors and external data metadata stay exact.

    max_gain clips the per-step guidance weight (the RTC paper's beta); 10 reproduces the original graph."""
    if not(isinstance(max_gain,(int,float))and math.isfinite(max_gain)and max_gain>0):raise ValueError('max_gain must be finite and positive')
    cap=np.float32(max_gain)
    g=model.graph;inputs={x.name for x in g.input};outputs=[x.name for x in g.output]
    shared=[f'__ensemble_member_{i}/hsr_action_chunk'for i in range(4)]
    if not ((inputs==BASE_INPUTS and outputs==['hsr_action_chunk'])or(inputs==BASE_INPUTS|{'second_noise'}and outputs==shared)):
        raise ValueError('Expected original or shared four-member learned interface')
    producer={};constants={x.name:nh.to_array(x)for x in g.initializer if x.data_location!=T.EXTERNAL}
    existing=inputs|{x.name for x in g.initializer}
    for i,n in enumerate(g.node):
        if n.op_type=='Constant' and len(n.output)==1:
            values=[a.t for a in n.attribute if a.name=='value']
            if len(values)==1:constants[n.output[0]]=nh.to_array(values[0])
        for x in n.output:
            if x in existing:raise ValueError('Duplicate graph value')
            existing.add(x);producer[x]=(i,n)
    if any(x.startswith('rtc_')or x.startswith('__rtc/')for x in existing):raise ValueError('RTC name collision')
    for name in ['action_mean','action_std']:
        a=constants.get(name)
        if a is None or a.shape!=(6,)or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Inline six-channel normalization required')
    if np.any(constants['action_std']<=0):raise ValueError('Positive action standard deviation required')
    seeds=inputs&{'initial_noise','second_noise'}
    seeds|={n.output[0]for n in g.node if n.op_type=='Neg'and len(n.input)==1 and n.input[0]in seeds}
    anchors={}
    def node(value,op):
        pair=producer.get(value)
        if pair is None or pair[1].op_type!=op:raise ValueError('Unexpected Euler/output topology')
        return pair
    for result in outputs:
        _,add=node(result,'Add')
        if 'action_mean'not in add.input:raise ValueError('Missing output mean')
        _,scale=node(next(x for x in add.input if x!='action_mean'),'Mul')
        if 'action_std'not in scale.input:raise ValueError('Missing output scale')
        _,sl=node(next(x for x in scale.input if x!='action_std'),'Slice');current=sl.input[0];chain=[]
        while current not in seeds:
            _,update=node(current,'Add');matches=[]
            for side,x in enumerate(update.input):
                pair=producer.get(x)
                if pair is None or pair[1].op_type!='Mul':continue
                j,mul=pair
                for k,c in enumerate(mul.input):
                    a=constants.get(c)
                    if a is not None and a.size==1 and a.dtype==np.float32 and float(a.item())==float(np.float32(-.1)):
                        matches.append((j,update.input[1-side],mul.input[1-k],1-k))
            if len(matches)!=1 or len(chain)>=10:raise ValueError('Require exactly ten fixed100ms Euler denoising steps')
            chain.append(matches[0]);current=matches[0][1]
        if len(chain)!=10:raise ValueError('Require exactly ten Euler denoising steps')
        for step,anchor in enumerate(reversed(chain)):
            j,previous,velocity,slot=anchor
            if j in anchors:raise ValueError('Unexpected shared noise-dependent Euler update')
            anchors[j]=(step,previous,velocity,slot)
    out=copy.deepcopy(model);q=out.graph;original=list(q.node);del q.node[:]
    added=[]
    def constant(name,value):
        value=np.asarray(value);added.append(h.make_node('Constant',[],[name],name=name,value=nh.from_array(value)))
        return name
    def op(kind,args,name,**attrs):
        added.append(h.make_node(kind,args,[name],name=name,**attrs));return name
    prefix=op('Sub',['rtc_prefix','action_mean'],'__rtc/center')
    prefix=op('Div',[prefix,'action_std'],'__rtc/normalized')
    pads=constant('__rtc/pads',np.array([0,0,0,0,0,26],np.int64))
    prefix=op('Pad',[prefix,pads],'__rtc/padded')
    q.node.extend(added);added.clear()
    for j,n in enumerate(original):
        if j in anchors:
            step,previous,velocity,slot=anchors[j];base=f'__rtc/step_{j}/'
            time=1-step/10;t=constant(base+'time',np.float32(time));tau=np.float32(1-time)
            if tau==0:gain=cap
            else:
                one=np.float32(1)-tau;squared=one*one
                gain=np.minimum(cap,(one/tau)*((squared+tau*tau)/squared))
            weight=constant(base+'gain',np.float32(gain))
            tv=op('Mul',[t,velocity],base+'tv');estimate=op('Sub',[previous,tv],base+'estimate')
            error=op('Sub',[prefix,estimate],base+'error');error=op('Mul',[error,'rtc_weights'],base+'weighted')
            correction=op('Mul',[error,weight],base+'correction')
            guided=op('Sub',[velocity,correction],base+'guided')
            selected=op('Where',['rtc_enabled',guided,velocity],base+'selected')
            q.node.extend(added);added.clear();n.input[slot]=selected
        q.node.append(n)
    q.input.extend([h.make_tensor_value_info('rtc_prefix',T.FLOAT,[1,16,6]),h.make_tensor_value_info('rtc_weights',T.FLOAT,[1,16,1]),h.make_tensor_value_info('rtc_enabled',T.BOOL,[])])
    meta=out.metadata_props.add();meta.key='rtc.guidance';meta.value=f'lerobot-guided-stop-gradient-identity-v1; max{float(cap):g}; tenEulerSteps'
    assert [x.SerializeToString()for x in g.initializer]==[x.SerializeToString()for x in q.initializer]
    return out,{'members':len(outputs),'guidedEulerSteps':len(anchors),'unchangedInitializers':len(g.initializer),'fullDenoiserJacobian':False,'weightsChanged':False,'policyInferenceCalls':0}
