# Experiment Status for the Manuscript

## Results Already Reported by Completed Runs

Full multi-task run:

- 288 real trajectories and 885 logged steps.
- Three authors, 16 tasks, two repetitions, and three watermark strengths.
- Author accuracy: 1.000.
- Exact timestamp accuracy: 0.003472.
- Timestamp-bucket accuracy: 1.000.
- Mean confidence: 1.000.
- Mean top-two margin: 0.275634.
- Abstention rate: 0.000.
- Strict task success: 0.96875.

Real-author scaling and stealth run:

- Maximum real authors: 20.
- Full decoder accuracy: 0.975.
- Accuracy at 20 real authors: 0.975.
- Random-guess accuracy at 20 authors: 0.050.
- Accuracy at 20 candidate identities: 0.958333.
- Mean Jensen--Shannon divergence: 0.00039494.
- Mean absolute probability shift: 0.00913554.
- Top-action flip rate: 0.0183673.
- Counterfactual detector AUC: 0.990916.

The source artifacts are now available locally in `full_robustness_report/`,
`aaai_real_author_scaling/`, and `logs/`. All 288 full-experiment manifests and
all 160 scaling manifests match a local clean JSONL file. The two sets contain
885 and 490 valid JSONL steps respectively, with no parse errors or missing
core fields. The robustness directory additionally contains 5,184 attacked
JSONL variants.

## Required Before AAAI Submission

### P0 Experiment Harness Implemented; Results Pending

The repository now contains `experiments/evaluate_access_tiers.py`, which
evaluates four explicitly separated audit views: action sequence only, action
sequence plus candidate set, post-watermark probabilities, and full trusted
pre/post probability logs. Its threshold is calibrated on a disjoint split.
`experiments/generate_control_logs.py` creates real unknown-author and
lambda-zero unwatermarked controls for open-set evaluation. These utilities
have not yet been run on new API trajectories, so no access-tier, false
attribution, or open-set number may be reported in the manuscript yet.

1. Freeze checksums and an immutable archive for the imported raw logs, CSV
   tables, and figures used by the manuscript.
2. Run at least three to five independent seeds and report 95% confidence
   intervals, not only point estimates.
3. Evaluate false-positive rates on genuinely unwatermarked agents and unseen
   identities; calibrate thresholds on a disjoint validation split.
4. Repeat the completed attack matrix across independent seeds and add paired
   significance tests; current intervals are trajectory-level only.
5. Add cross-model and cross-framework evaluation with fresh trajectories.
6. Report action-only, candidate-set-only, and trusted-probability-log access
   tiers so the black-box claim is precisely scoped.
7. Optimize stealth against a held-out detector. The current AUC of 0.991 is a
   major limitation despite the small average probability shift.
8. Replace SHA-256 over a public payload with an explicit keyed PRF/HMAC and
   evaluate key secrecy, forgery, and collusion threats.
9. Report runtime, API cost, hardware/software versions, and all final
   hyperparameters required by the AAAI reproducibility checklist.
10. Populate the official reproducibility checklist only after the experiments
    and artifact archive are frozen.
