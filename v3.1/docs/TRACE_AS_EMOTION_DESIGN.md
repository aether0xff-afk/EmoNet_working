# Trace-As-Emotion Design

## 1. Position

The working thesis of v3.1 is:

> an emotion is not only a label such as anger, shame, guilt, or sadness. An emotion is a structured trace of appraisal, target, control, social orientation, affect intensity, and action tendency over time.

In this view, `trace` is not auxiliary information. It is the internal emotional state representation.

## 2. Difference From v4

v4 currently proves a useful engineering fact:

```text
episode trace helps generation on targeted episode-sensitive inputs
```

v3.1 asks the deeper research question:

```text
does trace space itself behave like an emotion space?
```

The distinction matters:

| Version | Main role of trace | Proof target |
|---|---|---|
| v4 | prompt conditioning payload | better targeted responses |
| v3.1 | emotion-state representation | structured, stable, manipulable emotion space |

## 3. Emotion-State Components

The current trace fields can be interpreted as emotional state axes:

| Field | Emotion-state role |
|---|---|
| `valence` | pleasant/unpleasant direction |
| `arousal` | activation or intensity |
| `target` | self, other, situation, unknown object of emotion |
| `control_state` | controllability, helplessness, agency |
| `social_orientation` | social direction of the episode |
| `preserve` | affective content that must remain visible |
| `avoid` | response patterns that would distort the emotion |
| `action_tendency` | impulse or behavioral orientation |
| `episode_label` | coarse episode class, not the whole emotion |

The important claim is that emotion is the configuration and trajectory of these axes, not a single label.

## 4. Expected Structure

If trace is an emotion representation, then:

1. Records with similar `target`, `control_state`, `social_orientation`, and `action_tendency` should be close in trace space.
2. `anger-at-other` should separate from `guilt/self-blame`, even if both have negative valence.
3. High arousal, other-targeted blame should show different response constraints from low control sadness.
4. Nearest neighbors in trace space should share appraisal/action properties more often than chance.
5. Changing trace fields should change generated response direction even when the stimulus text is held constant.

## 5. What Would Count As Evidence

Representation-level evidence:

- nearest-neighbor label consistency above baseline
- cluster purity for `target`, `control_state`, `social_orientation`, and `action_tendency`
- low intra-group distance and high inter-group distance
- stable clusters across bootstrap samples

Generation-level evidence:

- trace ablation reduces appraisal fidelity
- trace perturbation changes response affect direction
- trace-preserving prompts retain raw affect better than stimulus-only prompts

Human-level evidence:

- blind evaluators prefer trace-conditioned responses specifically on appraisal fidelity and raw affect preservation
- evaluators can infer intended episode state from generated responses more accurately under trace-conditioned generation

## 6. Key Risk

The largest risk is that current trace fields are still too symbolic and manually compressed. If so, they may help prompting but fail to form a robust learned emotional geometry.

That would not invalidate v4. It would mean v3.1 needs richer neural trace vectors, recurrent trajectories, or learned latent states rather than only structured labels.

