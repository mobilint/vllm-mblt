# Embedding and reranking MXQs

Pooling uses a separate V1 worker. It evaluates each request as one unpadded
sequence, without vLLM prefix caching or chunked prefill. A batch of API inputs
is processed sequentially on one NPU core. Generation models keep their existing
worker and scheduling behavior.

| Source checkpoint | Output | Pooling / score |
| --- | --- | --- |
| `intfloat/e5-small` | 384 dimensions | Mean, L2 normalization |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 dimensions | Last token, L2 normalization |
| `nvidia/Nemotron-3-Embed-1B-BF16` | 2048 dimensions | Bidirectional attention, mean, L2 normalization |
| `Qwen/Qwen3-Reranker-0.6B` | One relevance score | Softmax over the final `no`/`yes` logits |
| `BAAI/bge-reranker-v2-m3` | One relevance score | Sigmoid of the sequence-classification logit |

The server takes a **compiled package path**, not the original FP32/BF16 Hub
checkpoint. A source checkpoint alone does not contain an MXQ.

## Before you start

Serving requires a Mobilint Aries NPU with its driver and runtime installed.
Compilation requires a separate CUDA GPU environment with the Mobilint compiler
SDK; it does not require an NPU on the compiler host. The builds described here
were validated on Aries with one NPU core and a maximum of 512 input tokens.
Other hardware targets and longer sequence lengths require separate validation.

Clone this repository and follow the [installation instructions](../README.md#installation)
on the NPU host. Use a source revision containing this pooling implementation;
installing an older PyPI release does not add these model classes.

Choose one of these paths:

- **Use a compiled package:** obtain a compatible complete pooling package and
  its checksum from your model provider. Verify the checksum, extract the
  archive, and continue with the serving examples below. Compilation is not
  required on the serving host.
- **Build your own package:** obtain a compatible Mobilint compiler SDK and
  follow the compilation and validation steps below. The tested compiler version
  is a development build; this repository does not provide its installer or
  guarantee that it is available through a public package index.

This repository does not include the five models' MXQ packages or publish a
public download URL for them. For NPU setup, SDK access, or availability of
compatible compiled packages, contact
[tech-support@mobilint.com](mailto:tech-support@mobilint.com).
The original Hugging Face checkpoints linked below are source weights, not
ready-to-serve Mobilint packages. You do not need access to an internal server
or a company network to follow these instructions once you have the required SDK
or compiled package.

All commands below run from the repository root. Paths such as `artifacts/e5-small`
are directories you create locally. If compiling on another machine, transfer
the entire resulting package to that path on the NPU host.

## Compile

The tested Aries builds use `qbcompiler==1.3.1.dev1`, PyTorch
`2.7.1+cu128`, and Transformers `4.57.1` in a separate CUDA compiler environment.
Install the matching Mobilint SDK (including its graph implementation), `datasets`,
`safetensors`, and `huggingface-hub`. The SDK is supplied separately; this plugin
does not redistribute compiler wheels. Run all commands from this checkout with
`PYTHONPATH=.`. The artifact manifest records the installed versions and exact
source-model commit used for each build.

```bash
export PYTHONPATH=.
python tools/prepare_pooling_calibration.py --output build/calibration.jsonl

python tools/compile_pooling.py \
  --model intfloat/e5-small \
  --calibration build/calibration.jsonl \
  --target-device aries-rb --max-length 512 \
  --output artifacts/e5-small
```

Repeat with each source checkpoint in the table and a new output directory.
Use `--revision <HF-commit>` to reproduce a previous build. Calibration accepts
JSONL rows containing `text` for embeddings and `query`/`document` for rerankers.
The provided public corpus is a starting point; measure quality on the intended
domain before using a package in production. No customer documents are required.

The default `--inference-scheme single` builds one-core execution. Use `all` only
when other runtime layouts are required; it increases compilation time. Qwen
compilation uses SDPA attention so the SDK recognizes its causal mask, explicit
8-bit weights / 16-bit activations, and SpinR1/R2. The exported input table is
rotated by the matrix from that same build. Never replace it with the source
checkpoint's table. Nemotron also uses 16-bit attention activations; its rotary
table is calibrated at the full compiled sequence length. Its residual stream also uses
SpinR1/R2, with a matching rotated embedding table.

`--max-length` is the package's served sequence limit, including special tokens
and reranker instructions. Increasing it requires a new compile and validation;
the original Hub context window is not a promise about an MXQ built at 512 tokens.

The compile output contains:

- `model.mxq`: transformer computation and, for rerankers, classification head.
- `embeddings.safetensors`: CPU embedding module/table matching that graph.
- Tokenizer files and `config.json`: vLLM registration and sequence limits.
- `source_config.json`: original embedding-module configuration.
- `pooling.json`: source revision, task, limits, compiler versions, and hashes.
- `source_README.md` and available source license/notice files.

Keep all package files together. The intermediate `calibration/` directory is
not required for serving and can be omitted when distributing the package.
MXQs and weights are separate deliverables, not source files committed to Git.

## Serve embeddings

The tested runtime uses vLLM `0.11.2`, PyTorch `2.9.0`, Transformers `4.57.1`,
`mobilint-qb-runtime==1.4.0`, and `mblt-model-zoo==2.6.0`. Install this plugin
with `pip install -e .` in that environment. Set `VLLM_API_KEY` to an API key
you choose, for example `export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"`.
Keep the same environment variable in the client shell, then run:

```bash
vllm serve ./artifacts/e5-small --runner pooling \
  --served-model-name intfloat/e5-small --max-model-len 512 \
  --host 127.0.0.1 --api-key "$VLLM_API_KEY"

curl http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"intfloat/e5-small","input":["query: What is the capital of France?","passage: Paris is the capital of France."]}'
```

Set the API key in the shell before starting the server. For another model,
change the artifact directory and served model name. Runtime placement is
`--model-loader-extra-config '{"dev_no":0,"core":0}'`; cores 0–7 map to the two
Aries clusters. The worker does not support tensor/pipeline parallelism, LoRA,
multimodal inputs, or caller-supplied input embeddings.

Embedding input strings follow the source model's conventions. E5 uses `query: `
and `passage: `. Qwen retrieval queries use `Instruct: <task>\nQuery: <query>`;
documents are supplied without this instruction. Nemotron uses `query: ` and `passage: `, as described in its
[model card](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16). The endpoint does not guess whether an arbitrary
string is a query or a passage. Optional dimension reduction is available for
the Matryoshka embedding checkpoints, followed by normalization.

## Serve reranking

```bash
vllm serve ./artifacts/qwen3-reranker --runner pooling \
  --served-model-name Qwen/Qwen3-Reranker-0.6B --max-model-len 512 \
  --host 127.0.0.1 --api-key "$VLLM_API_KEY"

curl http://127.0.0.1:8000/v1/rerank \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Reranker-0.6B","query":"What is the capital of France?","documents":["Paris is the capital of France.","Whales are marine mammals."]}'
```

Qwen's system instruction and final assistant suffix are inserted by the vLLM
score-template hook. BGE uses the tokenizer's query/document pair encoding.
`/score` is also supported. In vLLM 0.11.2, the built-in API-key middleware
protects `/v1/*` only: the `/rerank`, `/v2/rerank`, and `/score` aliases require
authentication at a reverse proxy before exposing the server to a network.
The example uses the authenticated `/v1/rerank` alias and loopback binding.
Scores default to [0, 1]; when activation is disabled,
Qwen returns `yes_logit - no_logit` and BGE returns its raw scalar logit.

## Validate independently

Generate references on the compiler/GPU host, then compare on the NPU host:

```bash
PYTHONPATH=. python tools/pooling_reference.py artifacts/e5-small \
  --device cuda --output artifacts/e5-small/reference.json
PYTHONPATH=. python tools/validate_pooling.py artifacts/e5-small \
  --reference artifacts/e5-small/reference.json \
  --output artifacts/e5-small/validation.json
```

When generating references on a separate GPU host, copy `reference.json` into
the corresponding package on the NPU host before running validation.

Nemotron's reference model requires a separate Transformers >=5.5 environment.
Serving uses only the exported table and MXQ, and remains compatible with the
Transformers 4.x dependency required by vLLM 0.11.2.

To check the HTTP server, including concurrent batches, API-key enforcement,
and rejection of inputs above the compiled limit:

```bash
python tools/validate_pooling_api.py artifacts/e5-small \
  --url http://127.0.0.1:8000 --output artifacts/e5-small/api-validation.json
```

This command reads `VLLM_API_KEY` from the environment and assumes that the served
model name matches `source_model` in the package manifest.

The smoke check includes different sequence lengths and unrelated relevance
pairs. Its default thresholds are embedding cosine >=0.98 and absolute score
error <=0.05. Passing it is numerical integration evidence, not a retrieval
benchmark. Store the validation report with the distributed package.

Source model cards: [E5](https://huggingface.co/intfloat/e5-small),
[Qwen embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[Nemotron](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16),
[Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
[BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3).

## Package separately from source

After both validation reports are present and passing:

```bash
python tools/package_pooling.py artifacts/e5-small \
  --output artifacts/releases/e5-small.tar.gz
```

The packager verifies the compiler manifest's checksums, requires matching
successful numerical and API validation reports, and excludes build
intermediates. It writes an archive and a `.sha256` sidecar. Extract the complete
archive on the NPU host and point `vllm serve` at the extracted directory.
