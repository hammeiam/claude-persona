# First time setup
```bash
# starting from empty directory
uv init
uv venv
uv add marimo polars "transformers[torch]" openai tqdm
uv run marimo edit
```

# Running this project
```bash
uv run marimo edit persona.py
```