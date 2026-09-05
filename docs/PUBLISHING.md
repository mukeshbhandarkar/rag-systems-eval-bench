# Publishing checklist

Nothing in this repository has been uploaded. Review all content, metrics, links, and licensing before publishing.

## Release repositories

- Dataset: [Max00035/rag-systems-eval-benchmark](https://huggingface.co/datasets/Max00035/rag-systems-eval-benchmark)
- Static Space: [Max00035/rag-systems-eval-bench](https://huggingface.co/spaces/Max00035/rag-systems-eval-bench)

The root source repository can remain separate. Do not create additional repositories unless they serve a clear purpose.

Source repository: [mukeshbhandarkar/rag-systems-eval-bench](https://github.com/mukeshbhandarkar/rag-systems-eval-bench)

## Before publication

1. Confirm the root MIT license and dataset CC BY 4.0 attribution match the intended release ownership.
2. Review every original synthetic passage, question, reference answer, and relevance label.
3. Rebuild and validate `ragbench-v1`; confirm all checked-in summary values come from persisted runs.
4. Verify that the published cards and dashboard link to the source repository above.
5. Confirm no model weights, caches, tokens, private paths, or arbitrary run directories are staged.

## Dataset repository

Create an empty Dataset repository manually in the Hugging Face web UI. Copy the contents of `hf_dataset/` to its repository root. The JSONL files use simple records compatible with the Dataset Viewer. Review the rendered card and viewer before making the repository public.

## Static Space

Create a Static HTML Space manually in the Hugging Face web UI. Copy the contents of `hf_space/` to its repository root. Its README front matter selects `sdk: static` and `app_file: index.html`. The Space needs no secret, model, backend, or upgraded hardware.

Preview before upload:

```bash
uv run python -m http.server 8000 --directory hf_space
```

Open the local address printed by the server and verify `data/results.json` and `data/failures.json` load.

## Updating results

Rebuild corpus and annotations under a new coordinated version when content or chunking changes. Run the three retrieval strategies with identical settings, run the optional reference generation in its own process, evaluate persisted answers, then execute `scripts/build_dashboard_data.py`. Copy regenerated dataset and dashboard files to their respective repositories only after reviewing the diff.

The `hf` CLI may be used manually after review, but it is not a project dependency and no upload command is part of the automated workflow. Never place an access token in this repository.
