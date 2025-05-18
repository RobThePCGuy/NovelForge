### What is NovelForge?

NovelForge is a collection of two LangGraph‑powered pipelines:

| Script                  | Purpose                | One‑line summary                                                                                                                        |
| ----------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **`novel_writer.py`**   | Green‑field generation | Plans, drafts, edits and refines brand‑new novels from a handful of parameters.                                                         |
| **`novel_rewriter.py`** | Transform & polish     | Splits an existing manuscript into chapters, then loops a *Rewriter→Editor→Refiner→Author‑review* cycle until each chapter is approved. |

Both graphs run completely offline against local **Ollama** models, can resume from checkpoints, and (optionally) maintain long‑range memory in a **Chroma** vector store.

---

### Key features

* **Agent specialization** – planner, beat‑sheet generator, drafter, editor, writer‑refiner, consistency checker and author feedback loops.&#x20;
* **Config‑first philosophy** – everything is driven by YAML (`llm_settings`, `workflow_settings`, `vector_store_settings`, `rewrite_settings`).&#x20;
* **Resume‑safe** – graphs checkpoint to SQLite; add `--thread-id` + `--resume` to pick up exactly where you left off.&#x20;
* **Vector‑store context** – earlier chapters are sharded and queried so later agents stay consistent in tone, facts and style.&#x20;
* **CLI overrides** – change genre, tone, concept, setting or output format on the fly without touching config.&#x20;

---

### Quick start

```bash
git clone https://github.com/<you>/novelforge.git
cd novelforge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see below
ollama pull llama3:8b && ollama pull mistral:7b
```

---

### Configuration

`config.yaml` (writer) and `config_rewrite.yaml` (rewriter) follow the same high‑level shape:

````yaml
llm_settings:
  author_model: "llama3:8b"
  drafter_model: "llama3:8b"
  editor_model: "mistral:7b"
  # …
  temperature:
    drafter: 0.8
workflow_settings:
  output_directory: "novel_output"
  max_llm_retries: 2
vector_store_settings:
  use_vector_store: true
rewrite_settings:            # writer ignores this block
  input_novel_file: "original_novel.txt"
  chapter_delimiter: "## Chapter "
  overall_improvement_focus: |
    - Tighten pacing
    - Enrich sensory detail
``` :contentReference[oaicite:7]{index=7}

*Change any value, or override at the CLI with the flags shown under **Usage**.*

---

### Usage

#### 1 · Create a brand‑new novel
```bash
python novel_writer.py --config config.yaml \
  --genre "Space Opera" --tone "Humorous" --output-format md
````

Key flags (full list in `--help`): `--genre`, `--setting`, `--concept`, `--tone`, `--target-audience`, `--thread-id`, `--resume`.&#x20;

#### 2 · Rewrite an existing manuscript

```bash
# put your draft at the path named in config_rewrite.yaml
python novel_rewriter.py --config config_rewrite.yaml
```

Add `--thread-id my‑rewrite --resume` to continue an interrupted pass.&#x20;

---

### Output

* **Novel/Rewrite** – Markdown or plaintext saved in `workflow_settings.output_directory`.
* **Logs** – `novel_generation.log` / `novel_rewrite.log` with INFO‑level progress.&#x20;
* **Vector store** – chapter shards under `<output>/vector_store/` (writer) or `<output>/vector_store_rewrite/` (rewriter).

---

### Extending NovelForge

* **Swap models** by editing `llm_settings`.
* **Change the pipeline** – each agent is a discrete node; add or remove nodes in the `build_graph`/`build_rewrite_graph` functions.
* **Integrate OpenAI** – replace `ChatOllama` with `ChatOpenAI` and supply your API key.

---

### Roadmap

* Automatic cover‑art generation.
* Interactive “live author” mode (steer the story mid‑chapter).
* Web UI with progress visualisation.

---

### License

MIT – feel free to build, tweak and publish your own stories.

---

### Acknowledgements

Huge thanks to the **LangChain**, **LangGraph**, **Ollama**, **Sentence‑Transformers** and **ChromaDB** communities for the open‑source tooling that makes NovelForge possible.

---

**Happy forging!**
