"""One stateless four-member graph; share observations only within the same request.

No file access, policy execution, weight conversion, output averaging or publication.
"""
import copy,hashlib
from onnx import helper

INPUTS={'head_rgb','hand_rgb','language_tokens','language_mask','hsr_state','initial_noise'}
# A prefix-conditioned graph also takes the executor's committed goals; they are request inputs shared by every member.
PREFIX_INPUTS={'action_prefix','prefix_mask'}
RESERVED='__ensemble_'

def share_ensemble(model):
 g=model.graph;nodes=list(g.node);inputs={x.name for x in g.input};initial={x.name for x in g.initializer}
 if inputs not in (INPUTS,INPUTS|PREFIX_INPUTS) or len(g.output)!=1 or g.output[0].name!='hsr_action_chunk' or g.sparse_initializer or model.functions:raise ValueError('Expected original six-input stateless learned graph')
 mask={x:2 if x=='initial_noise'else 1 for x in inputs};mask.update({x:0 for x in initial});producer={};consumers={}
 for i,n in enumerate(nodes):
  if n.domain not in ('','ai.onnx')or n.op_type in {'RandomNormal','RandomNormalLike','RandomUniform','RandomUniformLike','Multinomial','Dropout'}or any(a.type in (5,10)for a in n.attribute):raise ValueError('Random, control-flow or unsupported operator domain')
  flag=0
  for x in n.input:
   if not x:continue
   if x not in mask:raise ValueError('Unbound or nontopological node input')
   flag|=mask[x];consumers.setdefault(x,[]).append(i)
  for x in n.output:
   if not x:continue
   if x in mask:raise ValueError('Duplicate graph value')
   producer[x]=i;mask[x]=flag
 if any(x.startswith(RESERVED)for x in mask)or 'second_noise'in mask:raise ValueError('Reserved ensemble value collision')
 if not mask.get(g.output[0].name,0)&2:raise ValueError('Policy output must depend on explicit noise')
 boundary={x for x in producer if not(mask[x]&2) and any(any(mask.get(y,0)&2 for y in nodes[j].output)for j in consumers.get(x,[]))}
 if not boundary:raise ValueError('No observation prefix can be shared')
 def ancestors(tensors,stop):
  selected=set();stack=list(tensors)
  while stack:
   x=stack.pop()
   if x in stop or x not in producer:continue
   i=producer[x]
   if i in selected:continue
   selected.add(i);stack.extend(v for v in nodes[i].input if v)
  return selected
 prefix=ancestors(boundary,set());suffix=ancestors([g.output[0].name],boundary)-prefix
 if any(mask[x]&2 for i in prefix for x in nodes[i].output if x):raise ValueError('Noise-dependent prefix')
 merged=copy.deepcopy(model);out=merged.graph;out.name=g.name+'-shared-four-member';del out.node[:];del out.output[:];del out.value_info[:]
 out.node.extend(copy.deepcopy(n)for i,n in enumerate(nodes)if i in prefix)
 out.node.extend([helper.make_node('Neg',['initial_noise'],[RESERVED+'negative_0'],name=RESERVED+'negate_0'),helper.make_node('Neg',['second_noise'],[RESERVED+'negative_1'],name=RESERVED+'negate_1')])
 second=copy.deepcopy(next(x for x in g.input if x.name=='initial_noise'));second.name='second_noise';out.input.append(second)
 for value in g.value_info:
  if value.name in producer and producer[value.name]in prefix:out.value_info.append(copy.deepcopy(value))
 member_noise=['initial_noise',RESERVED+'negative_0','second_noise',RESERVED+'negative_1']
 for member,noise in enumerate(member_noise):
  rename={x:RESERVED+f'member_{member}/'+x for i in suffix for x in nodes[i].output if x}
  for i,n in enumerate(nodes):
   if i not in suffix:continue
   clone=copy.deepcopy(n);clone.name=RESERVED+f'member_{member}/node_{i}/'+n.name
   for j,x in enumerate(n.input):clone.input[j]=noise if x=='initial_noise'else rename.get(x,x)
   for j,x in enumerate(n.output):clone.output[j]=rename.get(x,x)
   out.node.append(clone)
  result=copy.deepcopy(g.output[0]);result.name=rename[g.output[0].name];out.output.append(result)
  for value in g.value_info:
   if value.name in rename:v=copy.deepcopy(value);v.name=rename[value.name];out.value_info.append(v)
 assert all(a.SerializeToString()==b.SerializeToString()for a,b in zip(g.initializer,out.initializer))
 proof=dict(members=4,memberOrder=['n1','-n1','n2','-n2'],prefixNodes=len(prefix),suffixNodesPerMember=len(suffix),boundaryTensors=len(boundary),outputNames=[x.name for x in out.output],originalNodes=len(nodes),transformedNodes=len(out.node),unchangedInitializers=len(out.initializer),initializerMetadataSha256=hashlib.sha256(b''.join(x.SerializeToString()for x in g.initializer)).hexdigest(),crossRequestCache=False,policyInferenceCalls=0,rawNumericalEquivalenceUnverified=True,speedupUnmeasured=True)
 return merged,proof
