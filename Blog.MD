# Training an LLM to Behave More Like an On-Call SRE

When language models are asked to handle production incidents, they often fail in a very specific way: they sound reasonable while doing the wrong thing.

They acknowledge incidents repeatedly, apply generic mitigations without understanding the root cause, or resolve too early because the response looks operationally neat even though the system is still unhealthy. That gap between plausible language and useful action is what motivated Incident Ops OpenEnv.

This project was built for OpenEnv Hackathon 2026 under Theme 3, World Modeling, Sub-theme 3.1, Professional Tasks. The aim was to create an environment where an LLM is judged not on how persuasive it sounds, but on whether it makes better decisions inside a realistic operational workflow.

## Why this environment

Many agent benchmarks still revolve around games, static question-answering, or simplified tool loops. Those are useful, but they do not capture the ambiguity and sequencing pressure of a real incident. In production operations, the model has to decide what to inspect, what to ignore, when to mitigate, when to communicate, and when it is actually safe to declare recovery.

That makes incident response a strong setting for reinforcement learning. It is sequential, partially observable, and easy to get wrong in ways that are operationally meaningful.

The central question behind this environment is straightforward:

Can a small instruct model learn to act less like a confident autocomplete system and more like a careful on-call engineer?

## What the model sees and does

Incident Ops OpenEnv presents the model with a structured incident state at every step. The observation includes customer impact, active alerts, service health, metrics, and recent actions. From that state, the model must choose exactly one next action.

The action space is intentionally operational:

- diagnostics such as `inspect_db_metrics`, `inspect_deploy_history`, and `inspect_network_topology`
- mitigations such as `rollback_auth_deploy`, `scale_db_cluster`, `withdraw_bgp_route`, and `archive_old_logs`
- workflow actions such as `acknowledge_incident`, `post_status_update`, and `resolve_incident`

The intended behavior is not simply to take actions. It is to follow the right sequence:

`diagnose -> mitigate -> communicate -> resolve`

That sequencing matters because one of the most common failure modes in these settings is that the model reaches for a mitigation too early, or resolves an incident that is not actually stable.

## The six incident scenarios

To avoid overfitting the environment to a single pattern, the benchmark includes six deterministic scenarios with different failure modes:

| Task | Scenario | What it tests |
| --- | --- | --- |
| `easy` | Auth deploy regression | Whether the model can connect a bad deploy to a rollback |
| `medium` | Database saturation from traffic spike | Whether it can read metrics and apply capacity mitigation |
| `hard` | Cascading outage after auth deploy | Whether it can coordinate multi-step diagnosis and recovery |
| `network` | BGP route leak | Whether it can reason about network failures instead of defaulting to app-layer fixes |
| `memory_leak` | OOM kill restart loop | Whether it can infer service instability from runtime symptoms |
| `disk_full` | Log disk saturation | Whether it can recover from infrastructure pressure without violating operational constraints |

The goal was to make the environment broad enough that a shallow repeated pattern would not score well across tasks.

## Reward design

The reward function is designed to make low-quality operational behavior expensive.

It combines diagnosis quality, mitigation completion, recovery quality, communication quality, efficiency, and harmful-action penalties. On top of that, the GRPO setup adds auxiliary reward signals for:

- valid action formatting
- anti-loop behavior
- task alignment
- sequence ordering
- progress delta
- communication gating
- terminal outcome

This was important because the environment would be easy to game if all that mattered was eventually reaching a resolved state. The reward needs to distinguish between a model that actually investigated and recovered the incident and a model that stumbled into superficially safe output.

## Training setup

The training pipeline consists of:

1. SFT warm-start on optimal trajectories from all six tasks
2. GRPO fine-tuning against the live environment
3. artifact generation for evaluation and inspection
4. before/after comparison with `compare_inference.py`

The base model used for training was:

- `unsloth/Llama-3.2-1B-Instruct`

The full run was configured with:

- `GRPO_MAX_STEPS=300`
- `GRPO_PER_TASK_PROMPTS=8`
- `GRPO_MID_EPISODE_PROMPTS=60`

The project also includes a Colab notebook so the complete training path can be rerun without rebuilding the pipeline manually.

## What changed after training

The most important result is that training did not remain flat. The environment produced a measurable learning signal.

Across the six evaluation tasks, the trained policy outperformed the untrained baseline with an average score improvement of `+0.23`.

| Task | Base score | Trained score | Delta |
| --- | --- | --- | --- |
| `easy` | 0.35 | 0.36 | +0.01 |
| `medium` | 0.06 | 0.34 | +0.28 |
| `hard` | 0.15 | 0.27 | +0.12 |
| `network` | 0.00 | 0.38 | +0.38 |
| `memory_leak` | 0.00 | 0.38 | +0.38 |
| `disk_full` | 0.14 | 0.36 | +0.22 |
| `average` | 0.12 | 0.35 | +0.23 |

That is not the same thing as saying the policy is solved. It is not. Some rollouts remain noisy. The model can still repeat actions, stall, or choose low-value steps in the middle of an episode.

But the purpose of this environment was not to claim a perfect SRE agent. The purpose was to show that the environment is trainable, that the reward is meaningful enough to produce observable change, and that the resulting behavior moves in a better direction rather than a random one.

The training run also produced a clear upward trend in reward over time. Early behavior was unstable and often poor, while later behavior improved enough to separate the trained policy from the initial baseline. The resulting policy is still imperfect, but the training signal is strong enough to indicate meaningful learning rather than noise alone.

## Why this matters

Incident response is a strong benchmark for agent training because it forces the model to operate under uncertainty, use tools selectively, and resist the temptation to produce an operationally neat but incorrect answer.

If environments like this are useful, they create a path toward training models on workflows that look more like real work and less like closed-form benchmark tasks. Even modest gains matter here because they show that the environment is sensitive to learning and that the reward captures something operationally relevant.

Incident Ops OpenEnv is therefore meant to function as three things at once:

- a benchmark for professional tool use
- a training environment with measurable reward improvement
- a concrete example of how operational workflows can be turned into learnable environments

## Links

- Hugging Face Space: [incident-ops-openenv-final](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv-final)
- Hugging Face Blog: [Blog.MD](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv-final/blob/main/Blog.MD)
- Trained Model: [sre-agent-llama3-grpo](https://huggingface.co/chritsysajii/sre-agent-llama3-grpo)
- Open in Colab: [colab_training.ipynb](https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/master/colab_training.ipynb)
- GitHub Notebook: [colab_training.ipynb](https://github.com/Christy-Saji/incident-ops-openenv/blob/master/colab_training.ipynb)
