"""Portable emotional trajectories and editable activation-module settings."""
import copy,json,math,uuid,os,tempfile
from pathlib import Path
from datetime import datetime,timezone
FORMAT='affector-geometry-v1'
def now():return datetime.now(timezone.utc).isoformat()
def finite(v,lo=-1,hi=1):return type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi

def new(name='Untitled geometry'):
    return {'format':FORMAT,'id':str(uuid.uuid4()),'name':name,'created_at':now(),'updated_at':now(),
        'points':[{'t':0.,'v':-.6,'d':-.3,'i':.35},{'t':.5,'v':0.,'d':0.,'i':.75},{'t':1.,'v':.7,'d':.6,'i':.5}],
        'layers':{'8':1.,'16':1.,'23':.5},'controls':{'strength':.12,'inertia':.2,'axis_top_k':3,'axis_mass':1.,'axis_temperature':1.,'budget':.2},
        'basis':None,'provenance':'User-authored trajectory; uncalibrated activation map.'}

def validate(g):
    if not isinstance(g,dict) or g.get('format')!=FORMAT:raise ValueError('Not an AFFECTOR geometry.')
    uuid.UUID(g['id'])
    if not isinstance(g['name'],str) or not g['name'].strip():raise ValueError('A geometry needs a name.')
    pts=g['points']
    if not isinstance(pts,list) or not 1<=len(pts)<=100000:raise ValueError('Use 1–100,000 trajectory points.')
    last=-1.
    for p in pts:
        if not all(finite(p[k],0 if k in ('t','i') else -1,1) for k in ('t','v','d','i')) or p['t']<=last:raise ValueError('Points need increasing t in [0,1], valence/direction in [-1,1], intensity in [0,1].')
        last=p['t']
    if not isinstance(g['layers'],dict) or not g['layers']:raise ValueError('Select at least one transformer layer.')
    for k,v in g['layers'].items():
        if not k.isdigit() or not 0<=int(k)<24 or not finite(v,-2,2):raise ValueError('Layer IDs must be 0–23 and weights -2…2.')
    c=g['controls']
    for k,lo,hi in [('strength',0,2),('inertia',0,.99),('axis_top_k',1,3),('axis_mass',.01,1),('axis_temperature',.1,3),('budget',0,1)]:
        if not finite(c[k],lo,hi):raise ValueError('Invalid activation control: '+k)
    if type(c['axis_top_k']) is not int:raise ValueError('Axis top-k must be an integer.')
    b=g.get('basis')
    if b is not None:
        if not isinstance(b,dict) or not isinstance(b.get('vectors'),dict):raise ValueError('Invalid steering basis.')
        for k,rows in b['vectors'].items():
            if not k.isdigit() or not 0<=int(k)<24 or len(rows)!=3 or any(len(r)!=1024 or any(not finite(x,-1000,1000) for x in r) for r in rows):raise ValueError('Steering weights must be finite 3 × 1024 arrays per layer.')
    return copy.deepcopy(g)

def point(g,t):
    pts=g['points'];p=pts[0]
    if t<=p['t']:return [p['v'],p['d'],2*p['i']-1]
    for q in pts[1:]:
        if t<=q['t']:
            a=(t-p['t'])/(q['t']-p['t']);return [(1-a)*p[k]+a*q[k] if k!='i' else 2*((1-a)*p[k]+a*q[k])-1 for k in ('v','d','i')]
        p=q
    return [p['v'],p['d'],2*p['i']-1]

def import_atlas(data,name):
    if data.get('format')==FORMAT:return validate(data)
    g=new(name);rows=[];total=0
    if data.get('format')=='emotion-atlas-v1':rows=data['results'];total=len(data['paragraphs'])
    elif data.get('format') in ('emotional-atlas-object-v1','emotional-atlas-evolution-v2'):
        meta=data.get('metadata',{});rows=meta.get('snapshot_results',meta.get('analysis',[]));total=meta.get('total',len(rows))
        if not rows:raise ValueError('This object has no emotional analysis snapshot. Import the Atlas analysis session instead; mesh/PCA coordinates alone cannot identify emotional axes.')
    elif data.get('format') in ('emotional-atlas-transfer-v1','emotional-atlas-transfer-session-v1'):
        source=data['source'];total=source.get('total_units',source.get('total',0));rows=source.get('rows',[])
        if not rows:
            for k in source.get('knots',[]):rows.append({'id':k['id'],'valence':k['point'][0],'direction':k['point'][1],'intensity':(k['point'][2]+1)/2})
    else:raise ValueError('Open an AFFECTOR geometry or an Emotional Atlas session/export with emotional scores.')
    pts=[]
    for r in sorted(rows,key=lambda r:r['id']):
        if not all(finite(r.get(k),0 if k=='intensity' else -1,1) for k in ('valence','direction','intensity')):continue
        pts.append({'t':(r['id']-1)/max(1,total-1),'v':r['valence'],'d':r['direction'],'i':r['intensity']})
    if not pts:raise ValueError('No complete emotional coordinates; unknown direction is not imputed.')
    g['points']=pts;g['provenance']=f"Imported from {data['format']}; {len(pts)}/{total} complete points. Missing positions interpolated, endpoints clamped."
    return validate(g)

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as f:
            temp=f.name;json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,path);temp=None
    finally:
        if temp:Path(temp).unlink(missing_ok=True)

class Library:
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
    def save(self,g):
        g=validate(g);g['updated_at']=now();atomic(self.path/(g['id']+'.json'),g);return g
    def list(self):
        result=[]
        for p in self.path.glob('*.json'):
            try:result.append(validate(json.loads(p.read_text())))
            except (ValueError,KeyError,TypeError):continue
        return sorted(result,key=lambda g:g['updated_at'],reverse=True)
    def delete(self,identity):
        identity=str(uuid.UUID(identity));(self.path/(identity+'.json')).unlink()
