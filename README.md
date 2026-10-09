# Harness Policy Distillation (HPD)

**Harness Policy Distillation: Aligning Agent Policies with Runtime Semantics** turns an agent harness's runtime control state into explicit training supervision while preserving a native action policy that can run through the original harness at deployment.

- **Model:** [Migraine/Pi-Qwen3-8B on Hugging Face](https://huggingface.co/Migraine/Pi-Qwen3-8B)
- **Authors:** Junze Zhu, Weihao Chen, Zhiyang Zhou, Zhen Wu, Xinyu Dai

This repository contains the environment-independent algorithm components and tests. The manuscript is described here for context; manuscript source and generated paper artifacts are maintained separately.

## What HPD does

At each replayed model-call boundary, HPD replays a reference trajectory through the Pi AgentLoop, compiles the harness state into a privileged control document, samples candidate actions, and keeps candidates that satisfy the harness contract and reference-control requirements. One shared model is trained with three coupled views:

1. **Harness-Replay (T):** native history to the verified action target.
2. **Injected (I):** the control document is supplied while predicting the action.
3. **Runtime (R):** native history to the control document and runtime-state target.

The three views cycle in a 1:1:1 schedule. The control document is used during target construction and training; deployment supplies only the native context to the original harness.

## Main reported result

The paper evaluates Qwen3-8B with LoRA adaptation on 583 held-out TRAJECT-Bench tasks (2,140 Pi provider boundaries). The reported 60k-optimizer-step HPD checkpoint uses 20k three-view training rounds.

| Model | Task Success | Final-answer F1 | Step-SR | Runtime Compliance | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Base | 3.09 | 26.82 | 26.45 | 19.90 | 2.57 |
| Pi-Qwen3-8B (HPD) | **22.98** | **41.22** | **63.79** | **49.23** | **18.70** |

Scores are percentages. The paper also reports model-size and training-schedule studies, ablations, harness portability, and zero-shot transfer to 92 BFCL v4 cases without BFCL-specific retraining. On that transfer evaluation, full HPD reaches 21.74% task success.

## Algorithm components

- [`types.py`](hpd/types.py) defines harness-neutral trajectories, actions, observations, replay boundaries, and text examples.
- [`harness_replay.py`](hpd/harness_replay.py) defines the adapter contract, deterministic reference replay, and construction of the three supervision views.
- [`tir_training.py`](hpd/tir_training.py) validates complete T/I/R triplets, provides the balanced schedule, tokenization helper, completion-only loss, and a small optimizer loop.
- [`tests/`](hpd/tests/) contains deterministic replay and schedule tests.

Run the component tests with:

```bash
python -m unittest discover -s hpd/tests -v
```

The full Pi harness adapter, model loading, dataset preparation, distributed training, and end-to-end evaluation pipeline are still being organized for a later code upload.

## Model

The released checkpoint is [`Migraine/Pi-Qwen3-8B`](https://huggingface.co/Migraine/Pi-Qwen3-8B). It is a Qwen3-8B model adapted for tool-using agents in the Pi AgentLoop. Use it with the Pi harness, its tool schemas, lifecycle rules, and termination protocol; plain text generation does not reproduce the evaluation setting in the paper. Consult the model card for current loading instructions, base-model terms, and usage limitations.

## Citation

```bibtex
@article{zhu2026hpd,
  title   = {Harness Policy Distillation: Aligning Agent Policies with Runtime Semantics},
  author  = {Junze Zhu and Weihao Chen and Zhiyang Zhou and Zhen Wu and Xinyu Dai},
  year    = {2026},
  note    = {Public preprint}
}
```

## License and attribution

The algorithm components are released for research use. The Pi-Qwen3-8B weights follow the license and terms shown on their [Hugging Face model card](https://huggingface.co/Migraine/Pi-Qwen3-8B), and Qwen3 upstream terms apply to the base model. Check the licenses of the Pi harness and TRAJECT-Bench before redistributing their materials.
