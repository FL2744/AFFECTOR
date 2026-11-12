import sys,unittest,threading,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
import providers,atlas_analysis
class Attempts(unittest.TestCase):
 def test_local_invalid_assessment_is_not_retried(self):
  class Engine:
   calls=0
   def generate(self,*args,**kwargs):
    self.calls+=1;return {'text':'{}','tokens':1,'truncated':False}
  e=Engine()
  with self.assertRaises(atlas_analysis.AnalysisError):providers.evaluate('A passage.','Local Qwen','Qwen','',threading.Event(),e)
  self.assertEqual(e.calls,1)
 def test_api_transport_failure_is_not_retried(self):
  calls=[];original=atlas_analysis.secure_urlopen
  def fail(*a,**k):
   calls.append(1);raise urllib.error.HTTPError('https://example.invalid',503,'Unavailable',{},None)
  atlas_analysis.secure_urlopen=fail
  try:
   with self.assertRaises(atlas_analysis.RequestError):providers.evaluate('A passage.','OpenAI',atlas_analysis.MODEL,'test',threading.Event())
   self.assertEqual(len(calls),1)
  finally:atlas_analysis.secure_urlopen=original
if __name__=='__main__':unittest.main()
