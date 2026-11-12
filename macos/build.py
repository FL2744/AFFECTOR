"""Build a self-contained Apple Silicon app from the installed, relocated Qwen runtime."""
from pathlib import Path
import subprocess,plistlib,shutil,os
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('AFFECTOR_RUNTIME','/Users/will/VIBES/lamp-qwen/dist/LAMP-Qwen.app/Contents/Resources'))
APP=ROOT/'dist/AFFECTOR.app';CONTENTS=APP/'Contents';RES=CONTENTS/'Resources';MAC=CONTENTS/'MacOS'
RES.mkdir(parents=True,exist_ok=True);MAC.mkdir(exist_ok=True)
for name in ('Python.framework','Libraries','packages','licenses'):
 if not (RES/name).exists():subprocess.run(['/bin/cp','-cR',str(RUNTIME/name),str(RES/name)],check=True)
shutil.copytree(ROOT/'engine',RES/'engine',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
if not (RES/'engine/.hf').exists():subprocess.run(['/bin/cp','-cR',str(RUNTIME/'engine/.hf'),str(RES/'engine/.hf')],check=True)
for name in ('ATLAS_LICENSE','README.md'):
 if (ROOT/name).exists():shutil.copy2(ROOT/name,RES/name)
shutil.copy2(RUNTIME/'requirements-lock.txt',RES/'requirements-lock.txt')
plist={'CFBundleName':'AFFECTOR','CFBundleDisplayName':'AFFECTOR','CFBundleIdentifier':'edu.vt.l1001.affector','CFBundleExecutable':'AFFECTOR','CFBundlePackageType':'APPL','CFBundleShortVersionString':'0.1.0','CFBundleVersion':'1','LSMinimumSystemVersion':'26.0','NSHighResolutionCapable':True,'NSPrincipalClass':'NSApplication','CFBundleIconFile':'AFFECTOR.icns'}
(CONTENTS/'Info.plist').write_bytes(plistlib.dumps(plist))
subprocess.run(['swiftc','-parse-as-library','-O','-target','arm64-apple-macos26.0',str(ROOT/'macos/Affector.swift'),'-o',str(MAC/'AFFECTOR'),'-framework','SwiftUI','-framework','AppKit'],check=True)
if (ROOT/'artifacts/AFFECTOR.icns').exists():shutil.copy2(ROOT/'artifacts/AFFECTOR.icns',RES/'AFFECTOR.icns')
subprocess.run(['codesign','--force','--deep','--sign','-',str(APP)],check=True)
print(APP)
