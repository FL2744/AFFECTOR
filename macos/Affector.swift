import SwiftUI
import AppKit
import UniformTypeIdentifiers

func pretty(_ x: Any) -> String { guard let d = try? JSONSerialization.data(withJSONObject: x, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) else { return "{}" }; return String(decoding:d,as:UTF8.self) }
@MainActor final class Studio: ObservableObject {
    @Published var geometries: [[String:Any]] = []
    @Published var geometry: [String:Any] = [:]
    @Published var editor = ""
    @Published var showWeights = false
    @Published var source = ""
    @Published var output = ""
    @Published var report = ""
    @Published var status = "Starting AFFECTOR…"
    @Published var busy = false
    @Published var backend = "Local Qwen"
    @Published var model = "gpt-6-luna"
    @Published var key = ""
    @Published var evaluator = "Local Qwen"
    @Published var evalModel = "gpt-6-luna"
    @Published var evalKey = ""
    @Published var temperature = 0.0
    @Published var topP = 0.95
    @Published var topK = 40
    @Published var tokens = 512
    @Published var verification: [String:Any] = [:]
    @Published var hasRun = false
    @Published var support = ""
    @Published var error: String?
    private var process: Process?
    private var input: FileHandle?
    private var buffer = Data()
    func start() {
        let resources = Bundle.main.resourceURL!
        let p = Process(); let stdin = Pipe(); let stdout = Pipe(); let stderr = Pipe()
        p.executableURL = resources.appendingPathComponent("Python.framework/Versions/3.14/bin/python3.14")
        p.arguments = ["-u",resources.appendingPathComponent("engine/bridge.py").path]
        p.currentDirectoryURL = resources.appendingPathComponent("engine")
        var env = ProcessInfo.processInfo.environment
        env["PYTHONHOME"] = resources.appendingPathComponent("Python.framework/Versions/3.14").path
        env["PYTHONPATH"] = resources.appendingPathComponent("packages").path
        env["HF_HOME"] = resources.appendingPathComponent("engine/.hf").path
        env["PYTHONNOUSERSITE"] = "1"; env["PYTHONDONTWRITEBYTECODE"] = "1"; env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        p.environment = env; p.standardInput = stdin; p.standardOutput = stdout; p.standardError = stderr
        input = stdin.fileHandleForWriting; process = p
        stdout.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            if !d.isEmpty { Task { @MainActor in self?.receive(d) } }
        }
        stderr.fileHandleForReading.readabilityHandler = { h in _ = h.availableData }
        p.terminationHandler = { [weak self] p in Task { @MainActor in self?.busy = false; self?.status = "Engine stopped (\(p.terminationStatus)). Reopen AFFECTOR to restart." } }
        do { try p.run() } catch { self.error = error.localizedDescription }
    }
    func stopProcess() { input?.closeFile(); process?.terminate() }
    func receive(_ d:Data) {
        buffer.append(d)
        while let end = buffer.firstIndex(of:10) {
            let line = buffer[..<end]; buffer.removeSubrange(...end)
            guard let obj = (try? JSONSerialization.jsonObject(with:Data(line))) as? [String:Any] else { continue }
            let data = obj["data"] as? [String:Any] ?? [:]
            switch obj["event"] as? String {
            case "ready", "result":
                if let a = data["geometries"] as? [[String:Any]] { geometries = a; if geometry.isEmpty, let first = a.first { select(first) } }
                if let g = data["geometry"] as? [String:Any] { select(g) }
                if let s = data["support"] as? String { support = s }
                if let t = data["text"] as? String { source = t }
                if let t = data["input"] as? String { source = t }
                if let b = data["backend"] as? String { backend = b }
                if let m = data["model"] as? String { model = m }
                if let s = data["sampling"] as? [String:Any] { temperature = s["temperature"] as? Double ?? 0; topP = s["top_p"] as? Double ?? 0.95; topK = s["top_k"] as? Int ?? 40; tokens = s["max_tokens"] as? Int ?? 512 }
                status = data["message"] as? String ?? "Ready"
            case "geometry":
                select(data)
                if let i=geometries.firstIndex(where:{($0["id"] as? String)==(data["id"] as? String)}) {geometries[i]=data}
            case "progress": status = obj["message"] as? String ?? "Working…"
            case "run": output = obj["output"] as? String ?? ""; report = displayReport(data); verification = data["verification"] as? [String:Any] ?? [:]; hasRun = true
            case "error": error = obj["message"] as? String; status = error ?? "Error"
            case "saved": status = "Transfer checkpoint saved."
            case "idle": busy = false
            default: break
            }
        }
    }
    func select(_ g:[String:Any]) { geometry = g; showWeights = false; var display=g; display.removeValue(forKey:"basis"); editor = pretty(display) }
    func toggleWeights() {
        guard var g=edited() else {return}
        showWeights.toggle()
        if !showWeights {geometry["basis"]=g["basis"];g.removeValue(forKey:"basis")}
        editor=pretty(g)
    }
    func displayReport(_ data:[String:Any])->String {
        var d=data
        if var g=d["geometry"] as? [String:Any] {g["basis"]="Compiled weights included in exported JSON; edit in Geometry + weights.";d["geometry"]=g}
        let result=pretty(d)
        return result.count > 80000 ? String(result.prefix(80000))+"\n… Export the report to view all passages." : result
    }
    func edited() -> [String:Any]? {
        guard let d = editor.data(using:.utf8), var g = (try? JSONSerialization.jsonObject(with:d)) as? [String:Any] else { error = "Geometry must be valid JSON. Check commas and brackets."; return nil }
        if g["basis"] == nil {g["basis"] = geometry["basis"] ?? NSNull()}
        return g
    }
    func send(_ q:[String:Any]) {
        guard let input, let d = try? JSONSerialization.data(withJSONObject:q) else { return }
        if q["op"] as? String != "cancel" && q["op"] as? String != "save_run" { busy = true }
        do { try input.write(contentsOf:d + Data([10])) } catch { busy = false; self.error = error.localizedDescription }
    }
    func open(_ op:String, document:Bool = false) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = document ? [UTType.plainText,UTType.rtf,UTType(filenameExtension:"docx")!] : [.json]
        if panel.runModal() == .OK, let url = panel.url { send(["op":op,"path":url.path]) }
    }
    func save(_ op:String, document:Bool = false) {
        let panel = NSSavePanel(); panel.nameFieldStringValue = document ? "transformed.txt" : (op == "export_geometry" ? "emotional-shape.json" : "affector-transfer.json")
        panel.allowedContentTypes = document ? [.plainText,.rtf,UTType(filenameExtension:"docx")!] : [.json]
        panel.allowsOtherFileTypes = false
        if panel.runModal() == .OK, let url = panel.url {
            var q:[String:Any] = ["op":op,"path":url.path]
            if op == "export_geometry" { guard let g = edited() else { return }; q["geometry"] = g }
            send(q)
        }
    }
    func transform() {
        guard let g = edited() else { return }
        send(["op":"transform","geometry":g,"text":source,"backend":backend,"model":model,"key":key,"sampling":["temperature":temperature,"top_p":topP,"top_k":topK,"max_tokens":tokens]])
    }
    func geometryAction(_ op:String) { if let g = edited() { send(["op":op,"geometry":g]) } }
}

struct Trajectory: View {
    @Environment(\.colorScheme) private var colorScheme
    let geometry:[String:Any]
    var body: some View {
        Canvas { ctx,size in
            let points = geometry["points"] as? [[String:Double]] ?? []
            func project(_ x:Double,_ y:Double,_ z:Double)->CGPoint { CGPoint(x:size.width*0.48 + CGFloat(x-y*0.45)*size.width*0.28,y:size.height*0.52 - CGFloat(z+y*0.35)*size.height*0.32) }
            let axes:[(String,(Double,Double,Double))] = [("X · valence",(1.35,0,0)),("Y · direction",(0,1.35,0)),("Z · intensity",(0,0,1.35))]
            for (name,a) in axes { var p=Path(); p.move(to:project(0,0,0)); p.addLine(to:project(a.0,a.1,a.2)); ctx.stroke(p,with:.color(.gray.opacity(0.6)),lineWidth:1); ctx.draw(Text(name).font(.system(size:10,design:.monospaced)),at:project(a.0,a.1,a.2),anchor:a.0 > 0 ? .trailing : .bottom) }
            var path=Path()
            for (i,p) in points.enumerated() { let pos=project(p["v"] ?? 0,p["d"] ?? 0,2*(p["i"] ?? 0.5)-1); if i==0 {path.move(to:pos)} else {path.addLine(to:pos)}; ctx.fill(Path(ellipseIn:CGRect(x:pos.x-3,y:pos.y-3,width:6,height:6)),with:.color(.primary)) }
            ctx.stroke(path,with:.color(.primary),lineWidth:2)
        }.frame(height:205).background(colorScheme == .dark ? Color.black : Color.white).overlay(Rectangle().stroke(.primary.opacity(0.15)))
    }
}
struct ContentView: View {
    @EnvironmentObject var s:Studio
    @AppStorage("appearance") private var appearance = "light"
    private var darkMode: Bool { appearance == "dark" }
    @State private var tab = 0
    @State private var confirmDelete = false
    let providers = ["Local Qwen","OpenAI","VT ARC"]
    var body: some View {
        VStack(alignment:.leading,spacing:0) {
            HStack(alignment:.top) {
                VStack(alignment:.leading,spacing:1) { Text("AFFECTOR").font(.system(size:53,weight:.black,design:.default)).tracking(-3); Text("EMOTIONAL GEOMETRY → WRITTEN FORM").font(.system(size:10,design:.monospaced)).tracking(2) }
                Spacer()
                VStack(alignment:.trailing,spacing:8) { Text(s.status).font(.system(size:12,design:.monospaced)).multilineTextAlignment(.trailing).frame(maxWidth:390,minHeight:35); if s.busy { Button("Stop") { s.send(["op":"cancel"]) } }
                    Picker("Appearance",selection:$appearance) {Text("Light").tag("light");Text("Dark").tag("dark")}.pickerStyle(.segmented).labelsHidden().frame(width:170).accessibilityLabel("Appearance")
                }
            }.padding(24)
            Divider().background(.primary)
            HStack(alignment:.top,spacing:0) {
                VStack(alignment:.leading,spacing:12) {
                    label("SHAPE LIBRARY")
                    ScrollView { VStack(alignment:.leading,spacing:3) { ForEach(s.geometries.indices,id:\.self) { i in
                        let g=s.geometries[i]
                        Button { s.select(g) } label: { VStack(alignment:.leading,spacing:5) { Text(g["name"] as? String ?? "Untitled").font(.system(size:13,weight:.medium)); Text(String((g["updated_at"] as? String ?? "").prefix(19))).font(.system(size:9,design:.monospaced)) }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background((g["id"] as? String)==(s.geometry["id"] as? String) ? Color.primary.opacity(0.10):Color.clear) }.buttonStyle(.plain).disabled(s.busy)
                    } } }.frame(minHeight:100,maxHeight:240)
                    HStack { Button("New") { s.send(["op":"new"]) }; Button("Open…") { s.open("import") } }.disabled(s.busy)
                    HStack { Button("Export…") { s.save("export_geometry") }; Button("Delete") { confirmDelete=true } }.disabled(s.busy || s.geometry.isEmpty)
                    Trajectory(geometry:s.geometry)
                    Text("Import Emotional Atlas sessions or object/evolution JSON with emotional coordinates. The line follows the text from beginning to end.").font(.system(size:11)).foregroundStyle(.secondary)
                    Spacer()
                    Text("LOCAL STEERING · API TRANSFER\nQWEN3.5 / 0.8B\nEXPERIMENTAL RESEARCH SOFTWARE").font(.system(size:9,design:.monospaced)).lineSpacing(4)
                }.padding(20).frame(width:255)
                Divider()
                VStack(alignment:.leading,spacing:16) {
                    Picker("Workspace",selection:$tab) { Text("01  Document").tag(0); Text("02  Geometry + weights").tag(1); Text("03  Verification").tag(2) }.pickerStyle(.segmented)
                    if tab==0 { document }
                    if tab==1 { geometry }
                    if tab==2 { verification }
                }.padding(22).frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.top)
            }
        }.background(darkMode ? Color.black : Color.white).foregroundStyle(.primary).preferredColorScheme(darkMode ? .dark : .light).frame(minWidth:1080,minHeight:770)
        .alert("AFFECTOR",isPresented:Binding(get:{s.error != nil},set:{if !$0 {s.error=nil}})) { Button("OK") {s.error=nil} } message: {Text(s.error ?? "")}
        .alert("Delete this geometry?",isPresented:$confirmDelete) { Button("Delete",role:.destructive) {s.send(["op":"delete","id":s.geometry["id"] ?? ""]);s.geometry=[:];s.editor=""};Button("Cancel",role:.cancel){} } message:{Text("Export a copy first if you want to retain its trajectory and calibrated weights.")}
    }
    func label(_ text:String)->some View { Text(text).font(.system(size:10,weight:.semibold,design:.monospaced)).tracking(1.5) }
    var document: some View {
        VStack(alignment:.leading,spacing:14) {
            HStack { Button("Open document…") {s.open("read_doc",document:true)}; Button("Open transfer…") {s.open("load_run")}; Button("Save transfer…") {s.save("save_run")}.disabled(!s.hasRun); Spacer();Button("Export output…") {s.save("export_doc",document:true)}.disabled(!s.hasRun) }.disabled(s.busy)
            HStack(alignment:.top,spacing:18) {
                VStack(alignment:.leading) {label("INPUT");TextEditor(text:$s.source).font(.system(size:15,design:.serif)).padding(6).overlay(Rectangle().stroke(.gray.opacity(0.4))).disabled(s.busy)}
                VStack(alignment:.leading) {label("TRANSFORMED TEXT");ScrollView {Text(s.output.isEmpty ? "Your transformed document will appear here." : s.output).font(.system(size:15,design:.serif)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(12)}.frame(maxWidth:.infinity,maxHeight:.infinity).overlay(Rectangle().stroke(.gray.opacity(0.4)))}
            }.frame(minHeight:190)
            Text("TXT · RTF · DOCX → text content. Document formatting is not preserved. Completed passages are saved automatically.").font(.system(size:10,design:.monospaced)).foregroundStyle(.secondary)
            Divider()
            HStack { Picker("Transform with",selection:$s.backend) {ForEach(providers,id:\.self){Text($0)}}.frame(width:260);Spacer();Button("Transform / resume") {s.transform()}.buttonStyle(.bordered).foregroundStyle(.primary).disabled(s.busy || s.geometry.isEmpty) }.disabled(s.busy)
            if s.backend != "Local Qwen" { HStack {TextField("Model",text:$s.model);SecureField("API key · kept in memory only",text:$s.key)}.textFieldStyle(.roundedBorder).disabled(s.busy);Text("API mode uses coordinate instructions. Hosted APIs do not expose transformer activations.").font(.system(size:11)) }
            else { HStack {Text("Sampling").font(.system(size:11,design:.monospaced));TextField("Temperature",value:$s.temperature,format:.number).frame(width:55);Text("temperature");TextField("Top-p",value:$s.topP,format:.number).frame(width:55);Text("top-p");TextField("Top-k",value:$s.topK,format:.number).frame(width:50);Text("top-k")}.font(.system(size:11)).disabled(s.busy) }
            HStack {TextField("Output tokens",value:$s.tokens,format:.number).frame(width:65);Text("maximum output tokens per passage (32–2048)").font(.system(size:11));Spacer();Button("Save progress…") {s.save("save_run")}.disabled(!s.hasRun)}
        }.onChange(of:s.backend) { _,b in if b=="VT ARC" {s.model="gpt-oss-120b"} else if b=="OpenAI" {s.model="gpt-6-luna"} }
    }
    var geometry: some View {
        VStack(alignment:.leading,spacing:12) {
            Text("A shape is a trajectory and a transformer module.").font(.system(size:24,weight:.semibold))
            Text("Edit the name, points, selected layers (0–23), controls, or compiled basis vectors below. Save to update the preview. Recalibrate after changing layers or model identity.").font(.system(size:12))
            Text("t = position · v = valence [−1,1] · d = direction [−1,1] · i = intensity [0,1]\nstrength = injection scale · budget = maximum relative RMS · inertia = temporal smoothing\naxis_top_k / axis_mass / axis_temperature select and reshape activation coefficients, independently of token sampling.").font(.system(size:11,design:.monospaced)).lineSpacing(5)
            HStack {
                control("Strength", "strength", 0...1)
                control("Budget", "budget", 0...1)
                control("Inertia", "inertia", 0...0.99)
            }.disabled(s.busy)
            TextEditor(text:$s.editor).font(.system(size:11,design:.monospaced)).padding(6).overlay(Rectangle().stroke(.gray.opacity(0.4))).disabled(s.busy)
            HStack {Button("Save geometry") {s.geometryAction("save")};Button("Calibrate local module") {s.geometryAction("calibrate")};Button(s.showWeights ? "Hide numerical weights" : "Edit numerical weights") {s.toggleWeights()};Spacer();Text("Calibration is experimental; emotional fit is measured separately.").font(.system(size:10)).foregroundStyle(.secondary)}.disabled(s.busy || s.geometry.isEmpty)
        }
    }
    func control(_ title:String,_ key:String,_ range:ClosedRange<Double>)->some View {
        let binding=Binding<Double>(get:{
            let obj=(try? JSONSerialization.jsonObject(with:Data(s.editor.utf8))) as? [String:Any]
            return (obj?["controls"] as? [String:Any])?[key] as? Double ?? 0
        },set:{ value in
            guard var g=(try? JSONSerialization.jsonObject(with:Data(s.editor.utf8))) as? [String:Any],var c=g["controls"] as? [String:Any] else {return}
            c[key]=value;g["controls"]=c;s.editor=pretty(g)
        })
        return VStack(alignment:.leading){Text("\(title)  \(binding.wrappedValue, specifier:"%.2f")").font(.system(size:11,design:.monospaced));Slider(value:binding,in:range).tint(.primary)}
    }
    var verification: some View {
        VStack(alignment:.leading,spacing:14) {
            Text("Measure the change, passage by passage.").font(.system(size:24,weight:.semibold))
            Text("Evaluate original and revised text with the Emotional Atlas scoring prompt. Similarity measures proximity to the requested coordinates; it is not a probability or a measure of semantic fidelity.").font(.system(size:12))
            HStack {Picker("Evaluator",selection:$s.evaluator){ForEach(providers,id:\.self){Text($0)}}.frame(width:260);Spacer();Button("Verify output") {s.send(["op":"verify","provider":s.evaluator,"model":s.evalModel,"key":s.evalKey])}.disabled(s.busy || !s.hasRun);Button("Export report…") {s.save("export_report")}.disabled(s.busy || !s.hasRun)}
            if s.evaluator != "Local Qwen" {HStack {TextField("Evaluator model",text:$s.evalModel);SecureField("Evaluator API key",text:$s.evalKey)}.textFieldStyle(.roundedBorder)}
            HStack(spacing:30) { score("ORIGINAL",s.verification["before_similarity"] as? Double);score("TRANSFORMED",s.verification["after_similarity"] as? Double);score("IMPROVEMENT",s.verification["improvement"] as? Double);VStack(alignment:.leading){label("PAIRED COVERAGE");Text("\(s.verification["paired_count"] as? Int ?? 0) / \(s.verification["total_count"] as? Int ?? 0)").font(.system(size:26,weight:.light))} }.padding(.vertical,12)
            Text("Similarity = 1 − distance / (2√3), with z = 2 × intensity − 1. Unknown directions and failed evaluations are excluded. Local 0.8B scoring may be unreliable; compare with an independent API evaluator.").font(.system(size:11)).foregroundStyle(.secondary)
            label("TRANSFER + EVALUATION RECORD · FULL WEIGHTS IN EXPORT")
            ScrollView {Text(s.report.isEmpty ? "Run a transfer, then verify its output." : s.report).font(.system(size:11,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(12)}.frame(maxWidth:.infinity,maxHeight:.infinity).overlay(Rectangle().stroke(.gray.opacity(0.4)))
        }.onChange(of:s.evaluator) {_,b in if b=="VT ARC" {s.evalModel="gpt-oss-120b"} else if b=="OpenAI" {s.evalModel="gpt-6-luna"} }
    }
    func score(_ name:String,_ value:Double?)->some View {VStack(alignment:.leading){label(name);Text(value.map{String(format:"%.3f",$0)} ?? "—").font(.system(size:26,weight:.light,design:.monospaced))}}
}
final class AppDelegate: NSObject, NSApplicationDelegate {
    var studio:Studio?
    func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {
        if studio?.busy == true {let a=NSAlert();a.messageText="Stop the active operation and quit?";a.informativeText="Completed passages are saved automatically. The current passage may need to be repeated.";a.addButton(withTitle:"Quit");a.addButton(withTitle:"Keep working");if a.runModal() != .alertFirstButtonReturn{return .terminateCancel}}
        studio?.stopProcess();return .terminateNow
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool {true}
}
@main struct AffectorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var studio=Studio()
    var body: some Scene {WindowGroup {ContentView().environmentObject(studio).onAppear{if delegate.studio == nil {delegate.studio=studio;studio.start()}}}.defaultSize(width:1230,height:850).commands {CommandGroup(replacing:.newItem){};CommandGroup(after:.saveItem){Button("Save transfer…"){studio.save("save_run")}.keyboardShortcut("s").disabled(!studio.hasRun)}}}

}
