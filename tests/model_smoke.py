import sys,threading,copy,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from steering import Engine,ActivationModule
import geometry,torch,providers
out=Path(__file__).resolve().parents[1]/'artifacts'
e=Engine(lambda s:print(s,flush=True));g=geometry.new('Validated smoke geometry');g['basis']=json.loads((out/'smoke-geometry.json').read_text())['basis'] if (out/'smoke-geometry.json').exists() else e.compile(g,threading.Event(),lambda s:print(s,flush=True));geometry.atomic(out/'smoke-geometry.json',g)
text='She opened the letter at the kitchen table. Her brother would arrive on Friday, and she needed to prepare the spare room.'
a=e.generate(text,max_tokens=160)
z=copy.deepcopy(g);z['controls']['strength']=0
b=e.generate(text,g=z,coordinates=[.8,.5,.5],max_tokens=160)
assert a['text']==b['text'],'Zero steering must preserve greedy output'
assert b['activation_injections']==0
g['controls']['strength']=.25
c=e.generate(text,g=g,coordinates=[.8,.5,.5],max_tokens=160)
assert c['activation_injections']>0
assert all(not p.requires_grad for p in e.model.parameters())
assert all(not l._forward_hooks and not l.post_attention_layernorm._forward_pre_hooks for l in e.layers)
d=e.generate(text,max_tokens=160);assert a['text']==d['text'],'Hooks must not affect later baseline'
module=ActivationModule(g['basis']['vectors']['8']);h=torch.randn(1,2,1024);delta=module(h,[1,1,1],g['controls'],2);assert float(delta.square().mean().sqrt())<=float(h[:,-1].square().mean().sqrt())*g['controls']['budget']+1e-5
try:score=providers.evaluate(c['text'],'Local Qwen','', '',threading.Event(),e)
except Exception as err:score={'error':str(err)}
geometry.atomic(out/'model-smoke.json',{'device':e.device,'identity':e.identity,'input':text,'baseline':a,'zero_strength':b,'steered':c,'local_evaluation':score,'checks':['zero-strength equivalence','nonzero activation injections','frozen parameters','hook cleanup','baseline restored','RMS budget']})
print(json.dumps({'baseline':a,'steered':c,'score':score},indent=2),flush=True)
