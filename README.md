# emotion2onnx

Extract emotions from audio. based on [emotion2vec](https://github.com/ddlBoJack/emotion2vec)

## Features

- Extract emotion from audio files

## Setup

```console
pip install -U emotion2onnx
```

- You also need [`emotion2vec.onnx`]()
- Please see examples

<details>

<summary>Instructions</summary>

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation) for isolated Python (Recommend).

Basically open the terminal (PowerShell / Bash) and run the command listed in their website.

_Note: you don't have to use `uv`. but it just make things much simpler. You can use regular Python as well._

2. Create new project folder (you name it)
3. Run in the project folder

```console
uv init -p 3.12
uv add emotion2onnx soundfile
```

4. Paste the contents of [`examples/save.py`](https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py) in `hello.py`
5. Download the files [`emotion2vec.onnx`]() and place it in the same directory.
6. Run

```console
uv run hello.py
```

That's it! emotions should be shown.

</details>

## Examples

See [examples](examples)

