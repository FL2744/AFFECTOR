import json,urllib.request,urllib.error,math
from atlas_analysis import secure_urlopen,PROMPT,SCHEMA,validate_results,analyze_batch,MODEL,RequestError,AnalysisError

def request(provider,model,key,system,text,max_tokens=1600):
    if not key.strip():raise ValueError('Enter the selected provider API key.')
    if provider=='OpenAI':
        url='https://api.openai.com/v1/responses';body={'model':model,'instructions':system,'input':text,'store':False,'max_output_tokens':max_tokens}
    elif provider=='VT ARC':
        url='https://llm-api.arc.vt.edu/api/v1/chat/completions';body={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':text}],'max_tokens':max_tokens,'stream':False}
    else:raise ValueError('Unknown provider.')
    try:
        req=urllib.request.Request(url,json.dumps(body).encode(),{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with secure_urlopen(req,timeout=150) as response:data=json.load(response)
        if provider=='OpenAI':
            if data.get('status')!='completed':raise ValueError('The API returned an incomplete response.')
            text=''.join(p.get('text','') for o in data.get('output',[]) for p in o.get('content',[]) if p.get('type')=='output_text')
        else:
            c=data['choices'][0]
            if c.get('finish_reason')!='stop':raise ValueError('The API response was truncated or refused.')
            text=c['message']['content']
        if not text.strip():raise ValueError('The API returned no text.')
        return {'text':text,'usage':data.get('usage',{}),'activation_injections':0,'mode':'prompt-guided API approximation; hidden states unavailable'}
    except urllib.error.HTTPError as e:raise ConnectionError(f'{provider} rejected the request (HTTP {e.code}). Check the model, key and account.') from None
    except OSError:raise ConnectionError(f'{provider} connection failed or timed out. Usage may have occurred.') from None

def evaluate(text,provider,model,key,cancel,engine=None):
    group=[{'id':1,'text':text,'context_before':'','context_after':''}]
    if provider=='Local Qwen':
        prompt=PROMPT+'\nReturn JSON only, matching this schema: '+json.dumps(SCHEMA)
        if cancel.is_set():raise InterruptedError('Verification stopped.')
        result=engine.generate(json.dumps({'paragraphs':group}),system=prompt,max_tokens=1800,cancel=cancel)
        usage={'local_tokens':result['tokens'],'attempts':1}
        if result['truncated']:raise ValueError('Local evaluator output was truncated.')
        raw=result['text'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        rows=validate_results(json.loads(raw),group)

    else:
        # Atlas's validated scoring adapter preserves its exact prompt and evidence checks.
        if provider=='OpenAI' and model!=MODEL:
            raw=request(provider,model,key,PROMPT+'\nReturn JSON only matching: '+json.dumps(SCHEMA),json.dumps({'paragraphs':group}),3000)
            rows=validate_results(json.loads(raw['text']),group);usage=raw.get('usage',{})
        else:rows,usage=analyze_batch(group,key,cancel,provider='arc' if provider=='VT ARC' else 'openai',model=model,max_attempts=1)
    return {'row':rows[0],'usage':usage}

def coordinate(row):
    return [row['valence'],row['direction'],2*row['intensity']-1] if row.get('direction') is not None else None

def metric(row,target):
    p=coordinate(row)
    if p is None:return None
    distance=math.dist(p,target)/(2*math.sqrt(3))
    return {'normalized_distance':distance,'similarity':1-distance,'coordinate':p,'target':list(target),'absolute_axis_errors':{'valence':abs(p[0]-target[0]),'direction':abs(p[1]-target[1]),'scaled_intensity':abs(p[2]-target[2])}}
