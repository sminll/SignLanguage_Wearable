# Server measurement snapshot / ?? ?? ???

This is a fixed snapshot, not a live server mirror. Original CSV fields and values are preserved. No model files are included.

## Layout

`ESP32|XIAO / YYYY-MM-DD / subjectN / session_id.csv`

See `manifest.json` for labels, counts, board, hand information and SHA-256 for every CSV. Subject numbers are collection identifiers; do not assume the same person across all devices/dates without confirmation. Dates use the original server timestamps (Korea local time).

## Training cautions

- `rest` rows are preserved for provenance; exclude them from sign-only training unless deliberately modeling rest.
- A CSV row is a sensor sample, not an independent gesture trial. Group by subject/session/trial; keep both hands together when splitting evaluation data. For new-person evaluation, split by person.
- The two health sessions are user-confirmed two-hand signs (`gesture_type=both` in manifest). Original CSV has no gesture-type/common-host-time fields. Do not fabricate synchronization or compare independent device boot clocks directly. Same-trial aggregation is exploratory; precise alignment is unverified.
- Subject2 retakes on Nov 2 have nearly zero FSR values and poor transfer accuracy. They are retained for review, NOT certified clean training data. See quality flags in manifest.
- Other files are also raw measurements, not a guarantee of training suitability. ESP32/XIAO and one-hand/two-hand measurements should be analyzed separately before combining.
- Test rows and user-deleted/withdrawn sessions are not exported. Calibration files elsewhere in `sensor-data/calibration` are separate and unchanged.
- Do not concatenate this snapshot with older ZIP exports: they overlap. This snapshot is not automatically included by existing training scripts; explicitly select the desired CSVs and review preprocessing.
