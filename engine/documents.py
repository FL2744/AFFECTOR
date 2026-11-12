"""Plain-content document import/export. Layout, tracked changes and media are not preserved."""
from pathlib import Path
import subprocess,tempfile,zipfile,xml.etree.ElementTree as ET,re
from xml.sax.saxutils import escape
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
def read(path):
    path=Path(path);ext=path.suffix.lower()
    if ext=='.txt':return path.read_text(encoding='utf-8-sig')
    if ext=='.rtf':return subprocess.check_output(['/usr/bin/textutil','-convert','txt','-stdout',str(path)]).decode('utf-8')
    if ext=='.docx':
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist())>100_000_000:raise ValueError('Document expands beyond the 100 MB import limit.')
            root=ET.fromstring(z.read('word/document.xml'))
        return '\n'.join(''.join(n.text or '' if n.tag==W+'t' else '\t' if n.tag==W+'tab' else '\n' if n.tag in (W+'br',W+'cr') else '' for n in p.iter()) for p in root.iter(W+'p'))
    raise ValueError('Choose .txt, .rtf or .docx.')
def write(path,text):
    path=Path(path);ext=path.suffix.lower()
    if ext=='.txt':path.write_text(text,encoding='utf-8');return
    if ext=='.rtf':
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'content.txt';src.write_text(text,encoding='utf-8')
            subprocess.run(['/usr/bin/textutil','-convert','rtf','-output',str(path),str(src)],check=True,capture_output=True)
        return
    if ext!='.docx':raise ValueError('Choose a .txt, .rtf or .docx filename.')
    body=''.join('<w:p><w:r><w:t xml:space="preserve">'+escape(line)+'</w:t></w:r></w:p>' for line in text.splitlines())
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+body+'<w:sectPr/></w:body></w:document>')
def spans(text,limit=700):
    out=[]
    for m in re.finditer(r'\S[\s\S]*?(?=\r?\n[ \t]*\r?\n|\Z)',text):
        start,end=m.span()
        while end>start and text[end-1].isspace():end-=1
        while end-start>limit:
            part=text[start:start+limit];cuts=list(re.finditer(r'[.!?؟。！？]["”’]*\s+|\s+',part));cut=cuts[-1].start() if cuts else limit
            if cut<limit//3:cut=limit
            stop=start+cut
            while stop>start and text[stop-1].isspace():stop-=1
            out.append((start,stop));start=stop
            while start<end and text[start].isspace():start+=1
        if start<end:out.append((start,end))
    return out

def assemble(text,locations,records):
    out=[];cursor=0
    for i,(a,b) in enumerate(locations):out.extend((text[cursor:a],records.get(str(i),{}).get('text',text[a:b])));cursor=b
    return ''.join(out)+text[cursor:]
