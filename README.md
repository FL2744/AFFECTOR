# AFFECTOR

<div align="left">

**Technical paper:** [PDF](paper/AFFECTOR.pdf) · [LaTeX source](paper/AFFECTOR.tex) · [Complete source ZIP](paper/AFFECTOR-source.zip) · [Markdown](paper/TECHNICAL_PAPER.md) · [Figures and source](paper/)

</div>

**Emotional geometry → written form.** A standalone native Mac research app for transferring an Emotional Atlas trajectory to a document, either through local transformer activation steering or through API instructions.

Created for William Taggart, Modern and Classical Languages and Literatures, Virginia Tech. Contact: wtaggart@vt.edu.

## Technical report

Read the [technical report](docs/TECHNICAL_REPORT.md) for the equations, implementation details, validation evidence and limitations.

This repository contains source, tests and documentation. The model weights, packaged application and numerical runtime are not included. The build script requires an existing relocated runtime; see the development notes below.

## Run a local build

Go [here](https://huggingface.co/will-taggart/AFFECTOR) to download the full macOS app. This 0.1.0 build bundles Qwen3.5-0.8B, Python, PyTorch and Transformers. No separate LAMP or Emotional Atlas installation is needed to run it. Requires **Apple Silicon and macOS 26+** because of the bundled numerical runtime. The app has a local ad-hoc signature; it has not been notarized for public distribution.

Use the **Light / Dark** control at the upper right to choose an appearance. Your choice is saved automatically between launches. Light is the initial default.

1. Open an Emotional Atlas JSON session or object/evolution export from **Shape library → Open**. Or select **New** and edit a trajectory in **Geometry + weights**.
2. Open a TXT, RTF or DOCX document, or paste text. Imports and exports preserve text content, not rich document formatting, tracked changes or images.
3. Choose **Local Qwen**, **OpenAI**, or **VT ARC**. API modes require your own key and model name. Keys stay in process memory and are excluded from geometry files, checkpoints and reports.
4. Choose **Transform / resume**. Local mode calibrates an activation basis on first use. Completed passages are saved automatically; use **Save progress** for a named checkpoint. Open that checkpoint to resume later. Failed or truncated revisions retain the original passage and record an error.
5. In **Verification**, choose an evaluator and **Verify output**. Export the full report to retain original/revised text, targets, measured coordinates, similarity, evidence, errors, model identity and settings.

The automatic checkpoint is `~/Library/Application Support/AFFECTOR/last-run.json`; geometry files live in the `geometries` subfolder. These files contain document text. Geometry import creates a separate library entry. Delete removes the selected library geometry; exported copies and saved transfers remain.

## Geometry and editable weights

A geometry JSON includes a UUID, distinct name, timestamps, trajectory points, layer weights, activation controls, calibration provenance and optionally a compiled basis. Use **Edit numerical weights** to reveal the compiled matrices; the usual editor keeps them out of the way. Strength, budget and inertia also have direct sliders. The editor exposes all values, including the actual 3 × 1024 numerical steering matrix at every selected layer. Saving validates ranges and dimensions. Recalibration replaces the basis with new contrastive measurements.

The [cthulhu-transfer.json](cthulhu-transfer.json) file in this repo can be used for testing. 

Trajectory points use normalized text position `t ∈ [0,1]`, valence `v ∈ [−1,1]`, direction `d ∈ [−1,1]`, and intensity `i ∈ [0,1]`. Atlas plot coordinates are `(v,d,2i−1)`. The trajectory is linearly interpolated, with endpoints clamped. Imported unsupported directions are omitted, never assigned a fabricated coordinate. Missing positions are interpolated explicitly. PCA meshes alone cannot recover these semantic axes; Atlas object exports must include their emotional-analysis snapshot.

Documents are split automatically into passages of at most 700 characters, preferring whitespace boundaries. A passage targets the trajectory at its original character midpoint. This is an approximation to a continuous emotional shape: every generated token in a passage receives the same target, with activation inertia across generation steps. It does not reproduce a solid mesh or guarantee exact emotional coordinates.

## Local transformer module

The frozen model is [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), pinned revision `2fc06364715b967f1860aea9cf38778875588b17`. Default selected layers are 8, 16 and 23 (zero-based).

Calibration captures pre-FFN states for four paired English contexts per emotional axis. The positive-minus-negative differences are averaged and RMS-normalized. Their rows form `B_l`, a 3 × 1024 editable matrix. This is an exploratory contrastive direction estimate, not a learned inverse of Emotional Atlas and not a validated universal emotional basis. The English calibration does not establish multilingual effectiveness.

For trajectory coordinate `c`, selected coefficients use `sign(c_j) |c_j|^(1/T)` with activation-specific top-k and cumulative-mass selection. The perturbation is scaled by user strength, layer weight, and the current hidden-state RMS; then smoothed against the previous step and capped by a relative RMS budget. At selected layers the block behaves as:

```
h' = h + delta
output = h' + FFN(norm(h'))
```

Hooks implement both the FFN input change and the residual change at the current final token, during prefill and subsequent decoding. No phrase, keyword, trigger or associative key is required. The model's base parameters remain frozen; hooks are removed in a `finally` block before subsequent generation or evaluation.

The design draws on LAMP/LAMP-Qwen's forward-only intervention and pre-FFN/residual placement. It does **not** load their associative key memories or claim their experiments validate emotional transfer. Module controls (`strength`, `inertia`, `axis_top_k`, `axis_mass`, `axis_temperature`, `budget`) affect hidden-state perturbations. Ordinary decoding temperature/top-p/top-k are separate; temperature zero uses greedy decoding. The local rewrite prompt does not contain the emotional target.

## API alternative

OpenAI uses the [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) with `store:false`. VT ARC uses its [OpenAI-compatible chat endpoint](https://docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html). Each receives the passage and desired emotional coordinates as prompt instructions. These APIs do not expose hidden states: API mode is a **prompt-guided approximation**, not activation steering. Activation and local sampling controls do not alter API inference. Output-token limits apply in both modes. API requests can incur charges/usage; selected passages are sent to the selected provider. No live authenticated API calls were made during development verification.

## Verification against Emotional Atlas

Original and revised passages are evaluated independently, with steering disabled, using the exact Atlas prompt, schema and evidence checks copied from Emotional Atlas. The evaluator can differ from the rewriting provider. Each assessment gets one attempt. Invalid responses, timeouts, connection failures and other chunk-level errors are flagged; the run continues to the next assessment. Failed rewrites retain the original chunk; failed assessments are excluded from similarity totals. Only an explicit Stop ends the run early. Model loading/calibration and checkpoint storage failures are run-level errors.

For target `c` and scored coordinates `p`:

```
normalized_distance = ||p − c||₂ / (2√3)
similarity = 1 − normalized_distance
improvement = mean(after_similarity) − mean(before_similarity)
```

The divisor is the full diagonal of the `[-1,1]³` Atlas cube. Similarity ranges from 0 to 1, is not a probability, and is meaningful only within the same coordinate interpretation. Means use paired valid before/after evaluations only. Coverage reports paired passages out of all document passages. Unknown direction excludes that score; error records and the complete coordinates/evidence are exported. Unchanged fallback passages are still visible and measurable, not hidden from the report.

These scores measure emotional proximity, **not semantic preservation**, literary quality, or factual accuracy. The 0.8B evaluator can give questionable interpretations even when schema and evidence validation pass. Use an independent stronger evaluator, inspect evidence, and compare multiple documents and trajectories before drawing research conclusions. Exact shape adherence is not guaranteed and there is no automatic claim of improvement.

## Validation and development

`tests/test_core.py` covers geometry validation/import, metric endpoints, TXT/RTF/DOCX round trips, whitespace-preserving assembly and library CRUD. `tests/model_smoke.py` runs the real local model and verifies zero-strength baseline equivalence, nonzero activation injections, frozen base parameters, hook removal, restored baseline generation and the RMS budget. Recorded development results are in `artifacts/model-smoke.json`; they are a smoke test, not a quality benchmark.

The native interface is SwiftUI/AppKit. A private subprocess communicates over JSON lines; there is no listening web server. Operations run off the UI thread. Local generation stops after a 120-second budget, checked between decoding steps; timed-out revisions retain the original and timed-out evaluations remain unscored. Stop interrupts local generation between decoding steps; an in-flight API request can take up to its timeout before stopping. API usage may already have occurred. Transfer exports carry the original source text, settings, weights, per-passage results and evaluation data, but no API key.

`macos/build.py` builds the bundle using a relocated runtime specified by `AFFECTOR_RUNTIME` (defaults to the installed LAMP-Qwen runtime on the development machine). It copies only runtime libraries, dependency licenses and model files, not LAMP memories. Swift compilation requires Xcode command-line tools. Runtime pins are in the bundled `requirements-lock.txt`. `run-engine.sh` uses the bundled runtime once built, with development overrides available.

Emotional Atlas scoring code is retained under `ATLAS_LICENSE`. The bundled Qwen model is Apache-2.0; dependency licenses are included in app Resources/licenses and package metadata. See [PyTorch hook documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html) for the intervention interfaces.
