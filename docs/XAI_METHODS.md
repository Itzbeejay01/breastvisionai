# Explainability and meta-learner validation

## Active baseline

The deployed model remains the four-input GradientBoostingClassifier. Its
ordered features are the selected CNN malignancy probabilities followed by
the equal-weight late-fusion probability. Runtime inference reads
n_features_in_ and, when present, feature_names_in_; it constructs exactly
that ordered schema and rejects an unsupported shape. The deterministic late
fusion input remains in the baseline for compatibility.

## SHAP interpretation

The baseline's late-fusion probability is the deterministic average of the
three CNN probabilities. Interventional TreeSHAP can vary this input separately
from its source values, creating inputs that cannot be produced by inference.
The TreeSHAP report now identifies the malignant estimator class explicitly.
The explanation pipeline validates a deterministic finite validation
background in the model's feature order. For the active four-feature baseline, it uses
tree-path-dependent SHAP in raw decision-score space. That avoids independently
perturbing the deterministic fusion duplicate; contributions can still divide
credit between correlated tree inputs. The raw reconstruction is checked
against decision_function. For a three-feature schema, probability-space
interventional SHAP is preferred; if unsupported, its original exception is
logged and the raw explanation is checked against decision_function. Raw scores
are never compared with probabilities.

Removing the exact redundant fusion feature gives a cleaner input definition,
but the three CNN probabilities remain correlated. Interventional SHAP still
does not model all such dependence and its contributions describe the fitted
model, not causal image evidence. Grad-CAM is also a coarse image attribution,
not lesion segmentation.

## Three-feature candidate

Run python model_training/train_three_feature_meta.py to fit a separate
three-CNN-probability candidate from validation predictions. It writes a
candidate artifact and ordered schema in Stacking_Result/; these files do
not replace the deployed four-feature artifact. The script reports validation
fit metrics and a single held-out test evaluation. The test numbers must not
be used to tune or select the candidate. Deployment requires separate review
of this evaluation and an explicit model-artifact change.

The current checkout contains cached prediction arrays and the saved
four-feature estimator, but does not include datasets_split/. As a result,
the cache labels' semantic association with the dataset directory names cannot
be independently re-audited from image folders here. The training scripts
document alphabetical flow_from_directory labels and name class 0 Benign and
class 1 Malignant; the saved estimator has numeric classes [0, 1]. Verify
the actual class directories and cache-generation provenance before treating
that mapping as independently confirmed.

## Sensitive local database

The tracked SQLite file contained authentication, upload, and prediction rows.
It has been removed from the repository and ignored. Start a clean local
database with python manage.py migrate; the deployment entrypoint performs
the same migration step. Existing rows migrated from an old database retain a
null owner and are visible only to authorized superusers. No Git history was
rewritten.
