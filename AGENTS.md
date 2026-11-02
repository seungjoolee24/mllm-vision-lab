# Research workspace rules

- Keep code portable between Windows, Linux and Colab. Never hard-code a personal drive path in the package or notebook.
- Keep models, downloaded datasets, generated runs, credentials and notebook execution outputs out of Git. The user explicitly chose code/notebooks/configurations only for GitHub. `history/` is local-only and must stay ignored, including Git commit history.
- Pin model revisions and dataset provenance. Never pass answer keys or reference CoT into model inputs.
- Record exact prompts, GPU, package versions, actual processed image sizes, visual/output tokens, seeds, batch membership, memory and time.
- In paired experiments, change only the declared treatment. On OOM, rerun all conditions in identical sub-batches and persist the split plan before retrying.
- Reports must directly show enough real question images, question text, options, ground truth and unabridged model responses. Separate observations, hypotheses and proposed follow-up tests. Report improvements and regressions together.
- Do not mix official scoring with unofficial normalized exact or manual semantic judgments.
- Run the CPU persistence tests and repository validator after relevant edits. GPU verification must be reported separately from Colab verification.
- Never claim a GitHub push, Drive authorization, GPU allocation or Colab inference succeeded unless verified.
