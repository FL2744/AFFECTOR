"""Private JSON-lines bridge for the native app. No listening network port."""
import sys,os,json,threading,copy,time,hashlib,traceback
from pathlib import Path
from geometry import Library,new,validate,import_atlas,atomic,point,now
import documents,providers
ROOT=Path(__file__).resolve().parent
SUPPORT=Path(os.environ.get('AFFECTOR_SUPPORT',str(Path.home()/'Library/Application Support/AFFECTOR')))
SUPPORT.mkdir(parents=True,exist_ok=True)
LIB=Library(SUPPORT/'geometries');LOCK=threading.RLock();OUTPUT=threading.Lock();cancel=threading.Event();engine=None;run=None;busy=False

def emit(event,**data):
    with OUTPUT:print(json.dumps({'event':event,**data},ensure_ascii=False,allow_nan=False),flush=True)
def progress(text):emit('progress',message=text)
def local():
    global engine
    if engine is None:
        from steering import Engine
        engine=Engine(progress)
    return engine

def state():return {'geometries':LIB.list(),'support':str(SUPPORT),'has_checkpoint':(SUPPORT/'last-run.json').exists()}
def save_run():
    with LOCK:
        if run:atomic(SUPPORT/'last-run.json',run)
def update():
    with LOCK:
        if run:emit('run',data=copy.deepcopy(run),output=documents.assemble(run['input'],run['spans'],run['records']))

def execute(q):
    global run
    op=q['op']
    if op=='list':return state()
    if op=='new':g=LIB.save(new(q.get('name','Shape '+now().replace('T',' ')[:19])));return {'geometry':g,**state()}
    if op=='save':g=LIB.save(q['geometry']);return {'geometry':g,**state()}
    if op=='import':
        data=json.loads(Path(q['path']).read_text(encoding='utf-8'));g=import_atlas(data,Path(q['path']).stem)
        # Import is a new library entry, never a silent replacement.
        import uuid
        g['id']=str(uuid.uuid4());g=LIB.save(g);return {'geometry':g,**state()}
    if op=='export_geometry':atomic(q['path'],validate(q['geometry']));return {}
    if op=='delete':LIB.delete(q['id']);return state()
    if op=='read_doc':return {'text':documents.read(q['path'])}
    if op=='calibrate':
        g=validate(q['geometry']);g['basis']=local().compile(g,cancel,progress);g=LIB.save(g);return {'geometry':g,**state()}
    if op=='load_run':
        data=json.loads(Path(q['path']).read_text())
        if data.get('format')!='affector-run-v1':raise ValueError('Not an AFFECTOR run checkpoint.')
        validate(data['geometry']);spans=documents.spans(data['input'])
        if [list(p) for p in spans]!=data['spans']:raise ValueError('Checkpoint passage layout does not match.')
        for k,r in data['records'].items():
            if not k.isdigit() or not 0<=int(k)<len(spans) or not isinstance(r.get('text'),str):raise ValueError('Invalid checkpoint records.')
        with LOCK:run=data
        update();return {'geometry':run['geometry'],'input':run['input'],'backend':run['backend'],'model':run['model'],'sampling':run['sampling']}
    if op=='export_doc':
        if run is None:raise ValueError('Transform a document first.')
        documents.write(q['path'],documents.assemble(run['input'],run['spans'],run['records']));return {}
    if op in ('export_report','save_run'):
        if run is None:raise ValueError('No transfer to save.')
        with LOCK:atomic(q['path'],run)
        return {}
    if op=='transform':
        g=validate(q['geometry']);text=q['text'];backend=q['backend'];model='Qwen/Qwen3.5-0.8B' if backend=='Local Qwen' else q['model'];sampling=q['sampling']
        if not text.strip():raise ValueError('Open a document or enter some text.')
        if not (0<=float(sampling['temperature'])<=2 and 0<float(sampling['top_p'])<=1 and 1<=int(sampling['top_k'])<=1000 and 32<=int(sampling['max_tokens'])<=2048):raise ValueError('Invalid sampling controls.')
        if backend=='Local Qwen':
            e=local()
            if g['basis'] is None or any(k not in g['basis']['vectors'] for k in g['layers']):
                g['basis']=e.compile(g,cancel,progress);g=LIB.save(g);emit('geometry',data=g)
        signature=hashlib.sha256(json.dumps([text,g,backend,model,sampling],sort_keys=True).encode()).hexdigest()
        if run is None or run.get('signature')!=signature:
            run={'format':'affector-run-v1','created_at':now(),'input':text,'geometry':g,'backend':backend,'model':model,'sampling':sampling,'signature':signature,'spans':[list(p) for p in documents.spans(text)],'records':{},'complete':False,'verification':{}}
        save_run()
        for index,(a,b) in enumerate(run['spans']):
            if cancel.is_set():raise InterruptedError('Stopped; completed passages saved.')
            if str(index) in run['records']:continue
            desired=point(g,(a+b)/max(1,2*len(text)));progress(f'Transforming passage {index+1}/{len(run["spans"])} · {backend}')
            try:
                if backend=='Local Qwen':result=e.generate(text[a:b],g=g,coordinates=desired,cancel=cancel,**sampling)
                else:
                    if backend not in ('OpenAI','VT ARC'):raise ValueError('Unknown backend.')
                    from steering import SYSTEM
                    instruction=SYSTEM+'\nThis is prompt-guided emotional shape transfer. Aim for valence %.3f, inward/outward direction %.3f, and intensity %.3f. These are semantic targets, not activation edits.'%(desired[0],desired[1],(desired[2]+1)/2)
                    result=providers.request(backend,model,q.get('key',''),instruction,text[a:b],max_tokens=sampling['max_tokens'])
                if not result['text'].strip():raise ValueError('Model returned empty output.')
                if result.get('truncated'):raise ValueError('Output hit the token limit; original retained. Increase output tokens for a new run.')
                result['target']=desired
            except Exception as error:
                if cancel.is_set():raise InterruptedError('Stopped; completed passages saved.') from error
                result={'text':text[a:b],'target':desired,'error':str(error),'error_type':type(error).__name__,'original_retained':True}
            with LOCK:run['records'][str(index)]=result
            save_run();update()
        run['complete']=True;save_run();update();return {'message':f"Transformation complete · {sum(bool(r.get('original_retained')) for r in run['records'].values())} passages retained after errors. Verify output to measure emotional fit."}
    if op=='verify':
        if run is None or not run['records']:raise ValueError('Transform at least one passage first.')
        provider=q['provider'];model='Qwen/Qwen3.5-0.8B' if provider=='Local Qwen' else q['model'];key=q.get('key','');e=local() if provider=='Local Qwen' else None
        scores={};paired=[]
        run['verification']={};save_run()
        for index,(a,b) in enumerate(run['spans']):
            record=run['records'].get(str(index))
            if not record:continue
            if cancel.is_set():raise InterruptedError('Verification stopped; completed scores retained.')
            progress(f'Verifying passage {index+1}/{len(run["spans"])} · {provider}')
            item={}
            for label,content in [('before',run['input'][a:b]),('after',record['text'])]:
                if cancel.is_set():raise InterruptedError('Verification stopped; completed scores retained.')
                progress(f'Verifying {label} text · passage {index+1}/{len(run["spans"])} · {provider}')
                try:
                    result=providers.evaluate(content,provider,model,key,cancel,e);item[label]=result;item[label+'_fit']=providers.metric(result['row'],record['target'])
                except Exception as error:
                    if cancel.is_set():raise InterruptedError('Verification stopped; completed scores retained.') from error
                    item[label+'_error']=str(error);item[label+'_error_type']=type(error).__name__
            if item.get('before_fit') and item.get('after_fit'):paired.append(item)
            scores[str(index)]=item
            run['verification']={'provider':provider,'model':model,'scores':scores,'paired_count':len(paired),'processed_count':len(scores),'total_count':len(run['spans']),'failed_assessments':sum(sum(k.endswith('_error') for k in item) for item in scores.values()),'complete':False,'metric':'similarity = 1 - Euclidean distance / (2 sqrt(3)); not a probability'}
            if paired:
                before=sum(i['before_fit']['similarity'] for i in paired)/len(paired);after=sum(i['after_fit']['similarity'] for i in paired)/len(paired)
                run['verification'].update(before_similarity=before,after_similarity=after,improvement=after-before)
            save_run();update()
        run['verification']['complete']=True;save_run();update();return {'message':f"Verification complete · {run['verification']['failed_assessments']} failed assessments flagged. Scores exclude failures and unknown directions."}
    raise ValueError('Unknown operation.')

def worker(q):
    global busy
    try:
        result=execute(q);emit('result',data=result)
    except Exception as error:
        save_run();emit('error',message=str(error))
    finally:busy=False;emit('idle')

def main():
    global busy
    if not (SUPPORT/'initialized').exists():
        if not LIB.list():LIB.save(new('From unease to possibility'))
        (SUPPORT/'initialized').touch()
    emit('ready',data=state())
    for line in sys.stdin:
        try:
            q=json.loads(line)
            if q['op']=='cancel':cancel.set();continue
            if q['op']=='save_run':
                with LOCK:result=execute(q)
                emit('saved',data=result);continue
            if busy:emit('error',message='Wait for the current operation or stop it first.');continue
            busy=True;cancel.clear();threading.Thread(target=worker,args=(q,),daemon=True).start()
        except Exception as error:emit('error',message=str(error))
if __name__=='__main__':main()
