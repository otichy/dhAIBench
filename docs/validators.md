# Validators

## Why Validators Exist

Some tasks have a very large label space, such as lemmatization across a full lexicon. In those cases it may be inefficient to put every valid label into every prompt.

The agent can delegate post-checks to an external NDJSON validator that can:

- accept a prediction, optionally normalizing it
- request a retry and provide a smaller `allowed_labels` set
- abort the run with a clear reason

## Protocol

- The agent starts the validator once per run.
- The agent sends one JSON object per attempt over stdin.
- The validator must return one JSON object on stdout.
- Validators must reserve stdout for protocol messages and write logs to stderr.

If `--validator_cmd` points to a `.py` file, the agent runs it with the current Python interpreter.

## Example: Lemmatization Validator

This repository includes `validators/lemmatization_validator.py`.

```bash
python benchmark_agent.py \
  --input data/input.csv \
  --model gpt-4o-mini \
  --validator_cmd validators/lemmatization_validator.py \
  --validator_args "--lexicon data/lemmata.txt --max_distance 2 --max_suggestions 30"
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
- `--validator_prompt_max_candidates`: cap rendered retry candidates
- `--validator_prompt_max_chars`: cap retry-instruction size
- `--validator_exhausted_policy`: choose the outcome when retries are exhausted
- `--validator_debug`: log raw NDJSON send/receive payloads at `DEBUG`

## Output Impact

When validation is enabled, the output CSV gains:

- `validatorStatus`
- `validatorReason`

Prompt-log entries also include validator metadata, with full request and response payloads in `full` prompt-log mode.
