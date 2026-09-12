# charvoice — Character Voice Generator

Turns a multi-character script into **one combined voiceover WAV**, each
character spoken in their own configured voice, in script order.

Full spec: [requirements.md](requirements.md).

## How it runs

- **Google Colab runs the pipeline.** Open [main.ipynb](main.ipynb) in Colab
  and run the cells top to bottom — that's the primary workflow.
- **Your machine runs the code, not the model.** The logic lives in
  [src/charvoice/](src/charvoice/) as an installable package with no GPU/torch
  dependency, so it can be edited, tested and linted locally.

## Local development

```bash
uv sync --extra dev
uv run pytest -q
```

Try the pipeline end to end with the built-in CPU `dummy` engine (no GPU, no
model needed):

```bash
uv run charvoice scripts/example.txt --overwrite
```

## Project layout

```text
src/charvoice/        the library — all logic lives here
main.ipynb            Colab entry point, thin driver only
voices/<id>/           one folder per character: profile.yaml + reference audio
config.yaml            central configuration
scripts/                script text files
output/                 generated voiceover.wav lands here
tmp/                    temp files and the segment cache
```

## Adding a character

1. Add reference audio under `voices/<id>/`.
2. Add `voices/<id>/profile.yaml` (copy an existing one as a template).
3. Use the character's name in the script.

No code changes required.
