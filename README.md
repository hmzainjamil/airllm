# AirLLM repository and research materials

> This repository contains an AirLLM source tree, model inference examples, and separate training and RLHF materials. The root content is a fork or copy with mixed provenance; verify source and license boundaries before reuse.
> **Status:** Source and notebooks present. Current compatibility, performance, and reproducibility have not been assessed here.

## Project identity

The nested `air_llm/` directory contains the AirLLM package source, a README, setup metadata, examples, and tests. The root also contains separate `examples/`, `training/`, `rlhf/`, `eval/`, and `data/` materials. These areas do not necessarily share the same upstream, dependencies, license, or validation status.

For upstream AirLLM documentation, see [the nested README](air_llm/README.md) and the [upstream project](https://github.com/lyogavin/airllm). The local copy may differ from upstream. No synchronization or fork parity is claimed.

## Repository map

| Path | Contents |
|---|---|
| `air_llm/` | Python package source, packaging metadata, notebooks, examples, and tests |
| `examples/` | Root-level inference notebook material |
| `training/` | Fine-tuning scripts and bilingual guides |
| `rlhf/` | DPO/RLHF scripts, data, and notes |
| `eval/` and `data/` | Evaluation notebooks and associated data |
| `LICENSE` | Apache License 2.0 text at this repository root |

This is a partial map of reviewed paths, not a complete inventory.

## Using the package

Consult [the nested AirLLM README](air_llm/README.md) for its upstream usage instructions and check them against the current source and your environment. The nested `air_llm/setup.py` identifies package name `airllm`, version `2.11.0`, and dependencies. Its install hook attempts to upgrade Transformers after installation, which can change the resolved environment. Review that behavior and dependency compatibility before installing.

Model loading may download model files from Hugging Face or access local files. Large models can require substantial storage, memory, and transfer time. Hardware requirements depend on model, dtype, backend, and workload. The repository's historical README performance statements are not validated by this fork's current checks.

No install, model download, inference run, training job, or test suite was run for this documentation update.

## License and provenance

There is a material license metadata conflict in the checked-in files: the root `LICENSE` contains Apache License 2.0 text, while the nested package setup metadata classifies the package as MIT. The upstream nested README also contains license badges and references that do not by themselves resolve this conflict. Do not assume the entire repository has one consistent license. A maintainer should establish the applicable license and preserve upstream notices before redistribution.

The nested README credits the AirLLM project at `lyogavin/airllm`; retain its notices and verify provenance for the additional root-level notebooks, training data, scripts, and assets.

## Safety and data handling

Treat downloaded model files, notebooks, datasets, and training scripts as external inputs. Inspect notebook cells and shell commands before running them. Model and dataset licenses and terms are separate from this repository's software license. Do not place private data in hosted inference, evaluation, or training workflows without checking where it is sent and stored.

## Security and support

No dedicated security reporting policy was found in the reviewed repository paths. Do not publish credentials or sensitive vulnerability details in public issues. Confirm a private maintainer channel before disclosure. This note is not a security audit.

## Validation status

- Repository tree, nested README, package setup metadata, and root license were reviewed.
- The Apache versus MIT metadata conflict is unresolved.
- No builds, tests, installs, downloads, inference, training, or evaluation runs were performed.
