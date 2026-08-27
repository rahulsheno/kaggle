### Approach

**Track A (restoration)** — the chain is given but not its parameters, so we run it forward
against the official corpus to manufacture as much matched (corrupted, clean) data as we
want. Two residual conv nets with different input views (raw de-salted crop + noise level,
and a contrast-normalized 4-channel view) are trained to invert the chain; their average is
the restored image. Per-crop noise level is estimated robustly (MAD of vertical differences
on the de-salted crop) and mapped to true sigma with a correction table fitted from simulated
corruptions, because clipping at the stored range truncates the noise tails. The epoch with
the best calibration-strip NRMSE is kept for each net.

**Track B (adjudication)** — two independent readers: one on the restored images, one
straight on the raw archive (the contrast-normalized view), each shift-averaged over small
translations, sharpened and blended with the blend picked on the bed-anchor digits. Both are
residual BatchNorm CNNs trained on a mix of lightly degraded corpus images and archive-chain
corruptions passed through the restoration ensemble, with epoch selection balancing anchor
accuracy against held-out simulated accuracy. The 128 bed-anchor digits, fixed by the bed
numbering, are the only labelled handwriting of the actual scribe and are used only for
selection and blending. Rows are then read against the ward/bed/formulary/dose constraints
and the check digit; the verdict is the row whose written digits admit the strongest
one-digit corruption explanation relative to the best legal closing, with the evidence
marginalised over every legal parse rather than committed to a single argmax read.

Everything below trains from the corpus fetched in Step 0. Seeded end to end; re-running the
notebook reproduces the submission.
