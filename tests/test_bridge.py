import sys,os,tempfile,unittest,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
_TMP=tempfile.TemporaryDirectory();os.environ['AFFECTOR_SUPPORT']=_TMP.name
import bridge,geometry,documents
bridge.emit=lambda *args,**kwargs:None
class Bridge(unittest.TestCase):
 def setUp(self):bridge.run=None;bridge.cancel.clear()
 def test_checkpoint_and_verification(self):
  original=bridge.providers.request;evaluate=bridge.providers.evaluate
  calls=[]
  def request(*a,**k):calls.append(a[4]);return {'text':'I felt deeply happy with my family.'}
  def score(text,*a,**k):return {'row':{'valence':.8 if 'happy' in text else -.4,'direction':.5,'intensity':.6},'usage':{}}
  bridge.providers.request=request;bridge.providers.evaluate=score
  try:
   g=geometry.new();g['points']=[{'t':0,'v':.8,'d':.5,'i':.6}]
   q={'op':'transform','geometry':g,'text':'I felt sad.\n\nI felt lonely.','backend':'OpenAI','model':'fake','key':'SECRET-DO-NOT-SAVE','sampling':{'temperature':0,'top_p':.95,'top_k':40,'max_tokens':128}}
   bridge.execute(q);self.assertEqual(len(calls),2)
   p=Path(_TMP.name)/'saved.json';bridge.execute({'op':'save_run','path':str(p)})
   self.assertNotIn('SECRET',p.read_text());bridge.run=None;bridge.execute({'op':'load_run','path':str(p)});bridge.execute(q);self.assertEqual(len(calls),2)
   bridge.execute({'op':'verify','provider':'OpenAI','model':'fake','key':'SECRET'})
   v=bridge.run['verification'];self.assertEqual(v['paired_count'],2);self.assertEqual(v['total_count'],2);self.assertEqual(v['after_similarity'],1);self.assertGreater(v['improvement'],0)
  finally:bridge.providers.request=original;bridge.providers.evaluate=evaluate
 def test_failed_unit_continues(self):
  original=bridge.providers.request;calls=[]
  def request(*a,**k):
   calls.append(1)
   if len(calls)==1:raise ValueError('Invalid rewrite')
   return {'text':'Second rewrite'}
  bridge.providers.request=request
  try:
   bridge.execute({'op':'transform','geometry':geometry.new(),'text':'First.\n\nSecond.','backend':'VT ARC','model':'fake','sampling':{'temperature':0,'top_p':.95,'top_k':40,'max_tokens':128}})
   self.assertTrue(bridge.run['records']['0']['original_retained']);self.assertEqual(bridge.run['records']['1']['text'],'Second rewrite');self.assertTrue(bridge.run['complete'])
  finally:bridge.providers.request=original
 def test_transport_interruptions_continue(self):
  original=bridge.providers.request;evaluate=bridge.providers.evaluate
  errors=[ConnectionError('offline'),TimeoutError('timeout'),InterruptedError('connection interrupted'),bridge.providers.RequestError('HTTP 503')]
  try:
   for error in errors:
    with self.subTest(error=type(error).__name__):
     bridge.run=None;calls=[]
     def request(*a,**k):
      calls.append(1)
      if len(calls)==1:raise error
      return {'text':'Second rewrite'}
     bridge.providers.request=request
     bridge.execute({'op':'transform','geometry':geometry.new(),'text':'First.\n\nSecond.','backend':'OpenAI','model':'fake','sampling':{'temperature':0,'top_p':.95,'top_k':40,'max_tokens':128}})
     self.assertEqual(len(calls),2);self.assertTrue(bridge.run['complete']);self.assertTrue(bridge.run['records']['0']['original_retained'])
     assessed=[]
     def assess(*a,**k):
      assessed.append(1)
      if len(assessed)==1:raise error
      return {'row':{'valence':.2,'direction':.5,'intensity':.6},'usage':{}}
     bridge.providers.evaluate=assess
     bridge.execute({'op':'verify','provider':'OpenAI','model':'fake'})
     v=bridge.run['verification'];self.assertEqual(len(assessed),4);self.assertTrue(v['complete']);self.assertEqual(v['failed_assessments'],1);self.assertEqual(v['paired_count'],1)
     self.assertIn('before_error',v['scores']['0']);self.assertIn('after_fit',v['scores']['1'])
  finally:bridge.providers.request=original;bridge.providers.evaluate=evaluate
 def test_explicit_stop_still_stops(self):
  original=bridge.providers.request;calls=[]
  def request(*a,**k):
   calls.append(1);bridge.cancel.set();raise InterruptedError('User stopped')
  bridge.providers.request=request
  try:
   with self.assertRaises(InterruptedError):
    bridge.execute({'op':'transform','geometry':geometry.new(),'text':'First.\n\nSecond.','backend':'OpenAI','model':'fake','sampling':{'temperature':0,'top_p':.95,'top_k':40,'max_tokens':128}})
   self.assertEqual(len(calls),1);self.assertFalse(bridge.run['complete']);self.assertEqual(bridge.run['records'],{})
  finally:bridge.providers.request=original;bridge.cancel.clear()
if __name__=='__main__':unittest.main()
