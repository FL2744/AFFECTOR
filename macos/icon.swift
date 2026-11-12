import AppKit
let dir=CommandLine.arguments[1]
for n in [16,32,64,128,256,512,1024] {
 let image=NSImage(size:NSSize(width:n,height:n))
 image.lockFocus()
 let size=CGFloat(n)
 NSColor.white.setFill();NSBezierPath(roundedRect:NSRect(x:size*0.04,y:size*0.04,width:size*0.92,height:size*0.92),xRadius:size*0.2,yRadius:size*0.2).fill()
 NSColor.black.setStroke();let p=NSBezierPath();p.lineWidth=size*0.04
 p.move(to:NSPoint(x:size*0.19,y:size*0.27));p.curve(to:NSPoint(x:size*0.79,y:size*0.7),controlPoint1:NSPoint(x:size*0.49,y:size*0.98),controlPoint2:NSPoint(x:size*0.38,y:size*0.02));p.stroke()
 for xy in [(0.19,0.27),(0.79,0.7)] {NSColor.black.setFill();NSBezierPath(ovalIn:NSRect(x:size*xy.0-size*0.04,y:size*xy.1-size*0.04,width:size*0.08,height:size*0.08)).fill()}
 image.unlockFocus()
 let rep=NSBitmapImageRep(data:image.tiffRepresentation!)!;let data=rep.representation(using:.png,properties:[:])!
 if n<=512 {try! data.write(to:URL(fileURLWithPath:"\(dir)/icon_\(n)x\(n).png"))}
 if n>=32 {try! data.write(to:URL(fileURLWithPath:"\(dir)/icon_\(n/2)x\(n/2)@2x.png"))}
}
