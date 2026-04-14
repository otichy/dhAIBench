# Validators

## Why Validators Exist

Some tasks have a very large label space, such as lemmatization across a full lexicon. In those cases it may be inefficient to put every valid label into every prompt.

The agent can delegate post-checks to an external NDJSON validator that can:

- accept a prediction, optionally normalizing it
- request a retry and provide a smaller `allowed_labels` set
- provide a custom retry message that is appended on top of the rebuilt base prompt
- abort the run with a clear reason

## Protocol

- The agent starts the validator once per run.
- The agent sends one JSON object per attempt over stdin.
- The validator must return one JSON object on stdout.
- Validators must reserve stdout for protocol messages and write logs to stderr.
- Retries rebuild the original prompt from scratch, then append the validator retry message as an extra user message.

If `--validator_cmd` points to a `.py` file, the agent runs it with the current Python interpreter.

In the bundled lemmatization validators, validator-side `--max_distance 0` disables the distance threshold. Returned candidates are still capped by the lexicon and any validator-side `--max_suggestions` limit.

## Example: Lemmatization Validator

This repository includes `validators/lemmatization_validator.py`.

```bash
python benchmark_agent.py \
  --input data/input.csv \
  --model gpt-4o-mini \
  --validator_cmd validators/lemmatization_validator.py \
  --validator_args "--lexicon data/lemmata.txt --max_distance 2 --max_distance_per_retry 1 --max_suggestions 30"
```

## Using `info` Metadata

The dataset `info` column is forwarded to the validator unchanged.

The reference lemmatization validator can use it to restrict candidates by part of speech:

- put POS into `info`, for example `pos=NOUN` or `part-of-speech:VERB`
- provide a lexicon with an optional second POS column
- choose the correct separator with `--lexicon_field_sep`

Example:

```bash
python benchmark_agent.py \
  --input data/input.csv \
  --model gpt-4o-mini \
  --validator_cmd validators/lemmatization_validator.py \
  --validator_args "--lexicon data/lemmata_with_pos.tsv --lexicon_field_sep tab --use_pos"
```

## Validator Flags

- `--validator_cmd`: enable validator-driven checking
- `--validator_args`: extra validator arguments as one quoted string
- `--validator_timeout`: timeout per validator roundtrip
- `--validator_prompt_max_candidates`: cap rendered retry candidates after any validator-side limit such as `--max_suggestions`
- `--validator_prompt_max_chars`: cap retry-instruction size
- `--validator_exhausted_policy`: choose the outcome when retries are exhausted
- `--validator_debug`: log raw NDJSON send/receive payloads at `DEBUG`
- validator-side `--max_distance_per_retry`: increase the validator threshold by a fixed amount starting with the second retry, so the third overall attempt is the first one with a higher threshold

Validator-side candidate limits such as `--max_suggestions` control how many labels the validator returns in `retry.allowed_labels`. The benchmark-side `--validator_prompt_max_candidates` then controls how many of those returned labels are actually rendered into the retry prompt.

## Output Impact

When validation is enabled, the output CSV gains:

- `validatorStatus`
- `validatorReason`

Prompt-log entries also include validator metadata, with full request and response payloads in `full` prompt-log mode.
