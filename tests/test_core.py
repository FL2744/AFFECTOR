import sys,unittest,tempfile,copy,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
import geometry,documents,providers
class Core(unittest.TestCase):
 def test_geometry(self):
  g=geometry.new();self.assertEqual(geometry.point(g,0),[-.6,-.3,-.30000000000000004]);self.assertEqual(geometry.point(g,.5),[0,0,.5])
  for bad in [float('nan'),3]:
   q=copy.deepcopy(g);q['points'][0]['v']=bad
   with self.assertRaises(ValueError):geometry.validate(q)
 def test_atlas(self):
  d={'format':'emotional-atlas-object-v1','metadata':{'total':3,'snapshot_results':[{'id':1,'valence':-.5,'direction':None,'intensity':.2},{'id':2,'valence':.1,'direction':.2,'intensity':.3},{'id':3,'valence':.3,'direction':.4,'intensity':.5}]}}
  g=geometry.import_atlas(d,'Test');self.assertEqual(len(g['points']),2);self.assertEqual(g['points'][0]['t'],.5)
 def test_metric(self):
  r={'valence':-1,'direction':-1,'intensity':0};self.assertEqual(providers.metric(r,[-1,-1,-1])['similarity'],1);self.assertAlmostEqual(providers.metric(r,[1,1,1])['similarity'],0)
  r['direction']=None;self.assertIsNone(providers.metric(r,[0,0,0]))
 def test_documents(self):
  text='First paragraph.\n\nA second line with 中文 and café.\nLast line.'
  with tempfile.TemporaryDirectory() as d:
   for ext in ['txt','rtf','docx']:
    p=Path(d)/('test.'+ext);documents.write(p,text);self.assertEqual(documents.read(p).strip(),text)
  for t in [text,'  Hello\r\n\r\nworld  ','word '*500,'中文'*1500]:
   spans=documents.spans(t);self.assertEqual(''.join(t[a:b] for a,b in spans).replace(' ','').replace('\r','').replace('\n',''),t.replace(' ','').replace('\r','').replace('\n',''));self.assertEqual(documents.assemble(t,spans,{}),t)
   records={str(i):{'text':t[a:b]} for i,(a,b) in enumerate(spans)};self.assertEqual(documents.assemble(t,spans,records),t)
 def test_library(self):
  with tempfile.TemporaryDirectory() as d:
   lib=geometry.Library(d);g=lib.save(geometry.new());self.assertEqual(lib.list()[0]['id'],g['id']);lib.delete(g['id']);self.assertEqual(lib.list(),[])
if __name__=='__main__':unittest.main()
