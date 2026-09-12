# Local Character Voice Generator — Requirements

## 1. Goal

Build a Google Colab-compatible application that accepts a scripted conversation/story containing multiple named characters and produces **one combined audio file** containing the complete voiceover in the correct order.

The system must support reusable character voice profiles so new characters can be added or existing voices can be replaced without changing the core pipeline.

Primary target use case:

1. Generate the script separately using Claude or another LLM.
2. Upload/paste the script into Google Colab.
3. The application identifies each character/speaker.
4. Each character's lines are synthesized using that character's configured voice.
5. All generated segments are combined in script order.
6. The user downloads one final audio file and adds it directly to a video editor.

---

## 2. Recommended Technology

### Runtime
- Google Colab
- Python 3.x
- GPU acceleration when available
- CUDA-compatible PyTorch

### Execution model: Colab notebook is the only entry point

All heavy work (model loading, TTS inference, audio encoding) runs on Colab's
GPU. The local machine is used only to write and version code. This is a hard
constraint: the local machine must never be required to run inference.

To make that work, the code is packaged as an installable Python library:

```text
src/charvoice/        <- all logic lives here (importable library)
main.ipynb            <- Colab notebook: thin driver, calls the library
```

Rules:

- `main.ipynb` contains **no business logic**. Every cell is a short call into
  `charvoice`. Parsing, validation, synthesis and assembly live in `src/`.
- The notebook installs the package from the Git repository, so Colab always
  runs the committed code. No copy-pasting code into cells.
- Notebook cells must be re-runnable. Re-running a cell must not duplicate
  state, reload an already-loaded model, or corrupt output.
- Anything that can be tested without a GPU (parser, profile loading,
  validation, config, cache keys, audio assembly maths) must be testable
  locally with no TTS model present.

### Packaging and dependency management

Two environments, one codebase:

| | Local machine | Google Colab |
|---|---|---|
| Purpose | write code, run fast tests | run the real pipeline on GPU |
| Manager | `uv` | `pip` |
| Torch / CUDA | not installed | already provided by Colab |
| Install | `uv sync --extra dev` | `pip install -e ".[colab]"` |

Reasoning: `uv` is used locally because it is fast and gives a reproducible
lockfile. `uv` is **not** used to build the Colab environment, because Colab
ships a pre-built Python with CUDA-enabled PyTorch already installed; creating
a fresh `uv` virtual environment there would discard that and force a
multi-gigabyte CUDA re-download on every session. In Colab, install into the
existing interpreter with `pip`.

Therefore:

- `torch` and other GPU/CUDA packages must be declared as **optional**
  dependencies, never as hard install requirements. Installing the package
  locally must never attempt to download CUDA wheels.
- Core dependencies must be CPU-safe and small (config parsing, audio I/O).
- The project must use a standard PEP 621 `pyproject.toml` with a `src/`
  layout, so `uv` and `pip` both install the exact same package.
- A `requirements.txt` may be generated from the lockfile for convenience, but
  `pyproject.toml` is the source of truth.

### Voice Generation
Primary engine:
- **GPT-SoVITS**

Alternative engines may be evaluated later:
- OpenVoice
- Other open-source local TTS/voice-conversion models

The voice engine must be isolated behind a small adapter/interface so it can be replaced without rewriting the rest of the application.

### Audio Processing
- FFmpeg
- Python audio library such as `soundfile` and/or `pydub`

### Configuration
- YAML or JSON
- Character voice profiles stored separately from application code

---

## 3. Input Script Format

The first version should support a simple, human-readable speaker format.

Example:

```text
NARRATOR: The city was completely silent that night.

SPIDER-MAN: Something feels wrong.

MILES: You feel it too?

SPIDER-MAN: Yeah. Stay close.

NARRATOR: Suddenly, a loud crash came from the alley.
```

Requirements:

- Speaker name must appear at the beginning of a line.
- Speaker names are case-insensitive for matching.
- Blank lines should be ignored.
- Preserve the original line order.
- Text after the first `:` is treated as the spoken content.
- Unknown speakers should produce a clear validation error rather than silently using the wrong voice.
- A future version may support `[CHARACTER]` tags or structured JSON input.

---

## 4. Character Voice Profiles

Voice profiles are a core requirement.

A profile represents the voice configuration for one character.

Example:

```yaml
characters:
  narrator:
    engine: gpt-sovits
    reference_audio: voices/narrator/reference.wav
    reference_text: "Reference transcript for the audio."
    language: en
    speed: 1.0

  spider-man:
    engine: gpt-sovits
    reference_audio: voices/spider-man/reference.wav
    reference_text: "Reference transcript for the audio."
    language: en
    speed: 1.0

  miles:
    engine: gpt-sovits
    reference_audio: voices/miles/reference.wav
    reference_text: "Reference transcript for the audio."
    language: en
    speed: 1.0
```

The exact fields should follow the selected TTS engine's requirements.

### Profile requirements

Each profile should be:

- Independently editable.
- Reusable across scripts.
- Easy to add without modifying application logic.
- Identified by a stable character ID.
- Able to specify its own reference audio/model configuration.
- Able to specify language.
- Able to specify optional speaking parameters such as speed.

The system should support an arbitrary number of profiles.

### Adding a new character

Adding a character should require only:

1. Add the reference voice/model files.
2. Add a profile entry.
3. Use the character's name in the script.

No Python source-code changes should be required.

---

## 5. Voice Profile Directory

Recommended structure:

```text
project/
├── src/
│   └── charvoice/            <- the installable library
│       ├── __init__.py
│       ├── config.py
│       ├── parser.py
│       ├── profiles.py
│       ├── audio.py
│       ├── cache.py
│       ├── pipeline.py
│       ├── colab.py          <- Colab-only helpers (Drive, upload, download)
│       └── engines/
│           ├── base.py
│           ├── registry.py
│           ├── dummy.py      <- CPU engine for testing without a GPU
│           └── gpt_sovits.py
│
├── main.ipynb                <- Colab entry point (thin driver only)
├── tests/
├── voices/
│   ├── narrator/
│   │   ├── profile.yaml
│   │   └── reference.wav
│   │
│   ├── spider-man/
│   │   ├── profile.yaml
│   │   └── reference.wav
│   │
│   └── miles/
│       ├── profile.yaml
│       └── reference.wav
│
├── scripts/
├── output/
├── pyproject.toml            <- source of truth for dependencies
├── uv.lock                   <- local reproducible environment
├── config.yaml
└── README.md
```

The implementation may adapt this structure to GPT-SoVITS requirements.

Note: a character's profile may live either in `voices/<id>/profile.yaml` or as
an entry in a single combined profiles file. Both must be supported, because
per-character files are easier to edit by hand while a combined file is easier
to carry around in Google Drive.

---

## 6. Processing Pipeline

The application should follow this pipeline:

```text
Script file
    |
    v
Script Parser
    |
    v
Speaker/Line Validation
    |
    v
Character Profile Resolver
    |
    v
TTS Engine
    |
    v
Generated Audio Segments
    |
    v
Audio Normalization
    |
    v
Audio Concatenation
    |
    v
Single Final Audio File
```

The final output should normally be:

```text
output/voiceover.wav
```

Optionally also support:

```text
output/voiceover.mp3
```

The WAV file should be the primary output for editing quality.

---

## 7. Audio Generation

For each script line:

1. Resolve the speaker.
2. Load that speaker's voice profile.
3. Send the line to the configured TTS engine.
4. Generate the audio segment.
5. Normalize technical audio properties where necessary.
6. Add the segment to the final timeline.
7. Insert configurable silence between lines if desired.

The system must maintain the exact script order.

---

## 8. Single-File Output

The default behavior must be a **single audio file**.

Example:

```text
output/
└── voiceover.wav
```

Individual temporary segments may be created internally, but they should not be the primary user-facing output.

Temporary files should be stored separately:

```text
tmp/
```

They may be deleted automatically after successful finalization.

---

## 9. Silence and Timing

Support configurable pauses.

Example:

```yaml
audio:
  pause_between_lines_ms: 250
  pause_between_scenes_ms: 800
```

At minimum, support:

- Default pause between dialogue lines.
- Optional pause after narrator lines.
- Optional explicit pause syntax in a future version.

The system should not add excessive silence automatically.

---

## 10. Audio Consistency

The final audio should have consistent:

- Sample rate
- Channel count
- Audio format
- Bit depth where applicable
- Overall volume level

All segments should be converted to a common format before concatenation.

Recommended final master format:

```text
WAV
Mono or stereo
44.1 kHz or 48 kHz
PCM
```

The exact format should be configurable.

---

## 11. Error Handling

The application should provide clear errors for:

- Missing character profile.
- Missing reference audio.
- Invalid script format.
- Empty spoken line.
- Unsupported language.
- Missing TTS model.
- GPU unavailable.
- TTS generation failure.
- Audio conversion failure.

Example:

```text
ERROR:
Character "Miles" appears in the script but has no voice profile.

Create:
voices/miles/profile.yaml
```

Do not silently fall back to another character's voice.

---

## 12. Google Colab Interface

`main.ipynb` is the primary and only user-facing entry point. It is a thin
driver: each cell is a short call into the `charvoice` library.

Recommended workflow:

### Cell 1
Check the GPU, clone the repository, and `pip install -e ".[colab]"`.
Do not create a virtual environment — install into Colab's own interpreter so
the pre-installed CUDA PyTorch is reused.

### Cell 2
Download/setup the selected TTS model, into a cached location so a re-run is a
no-op.

### Cell 3
Mount Google Drive or upload project files.

### Cell 4
Load config and voice profiles.

### Cell 5
Upload or paste the script.

### Cell 6
Validate the script. Report every unknown speaker at once, before any
synthesis starts.

### Cell 7
Generate the voiceover. The loaded engine is held in a module-level cache so
re-running this cell does not reload the model.

### Cell 8
Display:

```text
Generation complete.

Characters detected:
- Narrator
- Spider-Man
- Miles

Lines:
18

Output:
output/voiceover.wav
```

### Cell 9
Provide the final file for download.

A future version can replace the notebook workflow with a small web UI.

---

## 13. Google Drive Support

The project should optionally support Google Drive so voice profiles and models do not need to be uploaded every time.

Recommended structure:

```text
MyDrive/
└── character_voice_generator/
    ├── voices/
    ├── models/
    ├── scripts/
    └── output/
```

The Colab notebook should be able to use this directory as the persistent project directory.

---

## 14. Model Management

Do not hard-code model paths throughout the application.

Use configuration such as:

```yaml
tts:
  default_engine: gpt-sovits

  engines:
    gpt-sovits:
      model_dir: models/gpt-sovits
```

This makes it possible to change models later.

The system should distinguish between:

- TTS engine
- TTS model
- Character voice profile

This separation is important because multiple characters may use the same engine/model while having different voice references.

---

## 15. Architecture

Recommended components:

```text
ScriptParser
    |
CharacterResolver
    |
VoiceProfileManager
    |
TTSService
    |
AudioProcessor
    |
AudioAssembler
    |
OutputManager
```

### ScriptParser

Responsible for converting raw script text into structured lines.

Example internal representation:

```python
[
    {
        "speaker": "narrator",
        "text": "The city was completely silent that night."
    },
    {
        "speaker": "spider-man",
        "text": "Something feels wrong."
    }
]
```

### CharacterResolver

Maps script speaker names to voice profiles.

### VoiceProfileManager

Loads and validates character voice profiles.

### TTSService

Provides a unified interface to the selected TTS engine.

Example conceptual interface:

```python
audio = tts_service.generate(
    text="Something feels wrong.",
    voice_profile=spider_man_profile
)
```

The application should not depend directly on GPT-SoVITS throughout the codebase.

### AudioProcessor

Handles:

- Format conversion
- Resampling
- Volume normalization
- Silence generation

### AudioAssembler

Combines generated segments into one final audio timeline.

### OutputManager

Handles:

- Output naming
- WAV/MP3 export
- Temporary-file cleanup

---

## 16. Configuration

Use a central configuration file.

Example:

```yaml
project:
  name: character-voice-generator

tts:
  default_engine: gpt-sovits

audio:
  output_format: wav
  sample_rate: 48000
  channels: 1
  pause_between_lines_ms: 250

paths:
  voices: voices
  models: models
  scripts: scripts
  output: output
  temp: tmp
```

Avoid hard-coding these values.

---

## 17. Voice Profile Editing

The project must make profile editing easy.

At minimum, users should be able to change:

- Character name/ID
- Reference audio
- Reference transcript
- Language
- Voice model
- Speaking speed
- Optional pitch/style settings supported by the selected engine

Future UI should provide a voice-profile editor.

---

## 18. Voice Profile Safety

Only use reference voices that the user has permission to clone or synthesize.

The application should not assume that a voice reference is authorized merely because the user has the audio file.

---

## 19. Performance Requirements

The application should:

- Use GPU when available.
- Avoid repeatedly loading the same TTS model for every line.
- Cache loaded models where practical.
- Process lines sequentially by default to reduce VRAM pressure.
- Avoid loading every character model simultaneously if this exceeds GPU memory.
- Report generation progress.

Example:

```text
Generating 7/18
Speaker: Miles
```

For Google Colab, the implementation should work within typical free/paid Colab GPU constraints where possible.

### Performance rules that follow from the Colab-only model

Because a Colab session is billed by wall-clock time and dies when the runtime
is recycled, the expensive steps must not be repeated:

- **Load the model once per session.** The engine instance is held in a
  process-level cache keyed by engine name plus model configuration. Re-running
  the generate cell must reuse the loaded model, not reload it.
- **Group lines by speaker where the engine allows it**, so per-voice setup
  (reference audio encoding, prompt feature extraction) happens once per
  character instead of once per line. Output order must still follow the
  script exactly.
- **Cache reference-audio features per character**, not per line.
- **Keep the segment cache on Google Drive**, so it survives a runtime restart
  and speeds up script revisions across sessions.
- **Resolve and validate everything before loading the model.** Parsing,
  speaker validation and profile loading are cheap and must fail fast, before
  any GPU time is spent.
- **Stream segments to disk** rather than holding every segment in RAM, so long
  scripts do not exhaust the Colab instance's memory.
- **Free GPU memory between stages** where the engine supports it.
- Progress reporting must be incremental, so a disconnect still shows how far
  the run got.

---

## 20. Caching

Optional but recommended.

If the exact same:

- Text
- Voice profile
- Model configuration

is requested again, the system may reuse the generated segment.

Cache keys should change when the voice configuration changes.

This can significantly reduce generation time during script revisions.

---

## 21. Output Naming

Default output:

```text
voiceover.wav
```

Optionally derive the name from the script:

```text
my_spiderman_short_voiceover.wav
```

Avoid overwriting an existing output unless explicitly requested.

---

## 22. Minimum Viable Product

The first implementation should support only:

1. GPT-SoVITS.
2. Plain `CHARACTER: dialogue` scripts.
3. YAML/JSON character voice profiles.
4. Multiple characters.
5. One combined WAV output.
6. Google Colab GPU execution.
7. Basic pause insertion.
8. Basic audio normalization.
9. Clear validation errors.
10. Easy addition/editing of voice profiles.

Do not build a complex UI initially.

---

## 23. Future Features

Potential future additions:

- Web-based UI.
- Drag-and-drop script upload.
- Voice-profile management UI.
- Preview a character voice.
- Automatic speaker detection.
- Scene support.
- Emotion/style tags.
- SSML-like controls.
- Per-line speed.
- Per-line pitch.
- Background music.
- Sound effects.
- Automatic loudness mastering.
- Automatic subtitle generation.
- Automatic video assembly.
- Multiple TTS engines.
- Automatic model downloading.
- Local desktop version.
- API endpoint.
- Batch processing of multiple scripts.

These should not complicate the MVP.

---

## 24. Example End-to-End Usage

Input:

```text
NARRATOR: The streets were empty.

SPIDER-MAN: Did you hear that?

MILES: Hear what?

SPIDER-MAN: That sound behind us.

NARRATOR: Miles turned around slowly.
```

Profiles:

```text
narrator -> Narrator voice
spider-man -> Spider-Man voice
miles -> Miles voice
```

Processing:

```text
Line 1 -> Narrator TTS
Line 2 -> Spider-Man TTS
Line 3 -> Miles TTS
Line 4 -> Spider-Man TTS
Line 5 -> Narrator TTS
```

Final result:

```text
voiceover.wav
```

The final WAV contains the complete conversation in the same order as the script.

---

## 25. Development Priorities

Implement in this order:

### Phase 1
Project structure and configuration: `src/` layout, `pyproject.toml`, `uv`
lockfile, and a `main.ipynb` that installs the package and imports it.

### Phase 2
Script parser and validation.

### Phase 3
Voice profile manager.

### Phase 4
GPT-SoVITS adapter.

### Phase 5
Single-line voice generation test, first against the CPU dummy engine locally,
then against GPT-SoVITS on Colab.

### Phase 6
Multi-character generation.

### Phase 7
Audio concatenation and normalization.

### Phase 8
Colab upload/download workflow.

### Phase 9
Caching and progress reporting.

### Phase 10
Optional UI.

The core pipeline should remain usable from Python even if a UI is added later.

---

## 26. Definition of Done for the Colab Workflow

The project is working when all of the following hold:

1. A fresh Colab runtime can run `main.ipynb` top to bottom with no manual
   edits beyond choosing a script and a profiles file.
2. No cell in `main.ipynb` contains pipeline logic; each is a call into
   `charvoice`.
3. The local machine can install the package and run the full test suite with
   `uv sync --extra dev && uv run pytest`, with no GPU, no `torch`, and no TTS
   model present.
4. Parser, validation, profile loading, config merging, cache keys and audio
   assembly are all covered by tests that run locally.
5. Swapping the TTS engine requires only a config change plus one new adapter
   file under `src/charvoice/engines/`.
6. A second run over an unchanged script reuses cached segments and is
   noticeably faster.
