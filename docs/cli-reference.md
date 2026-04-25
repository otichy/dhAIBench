# CLI Reference

This file is generated from `python benchmark_agent.py --help` by `python scripts/generate_cli_reference.py`.

```text
usage: benchmark_agent.py [-h] [--input INPUT [INPUT ...]] [--labels LABELS]
                          [--output OUTPUT] [--resume] [--task_name TASK_NAME]
                          [--task_description TASK_DESCRIPTION] [--tags TAGS]
                          [--unclassified] [--repeat_unclassified]
                          [--model MODEL] [--temperature TEMPERATURE]
                          [--top_p TOP_P] [--top_k TOP_K]
                          [--service_tier {standard,flex,priority}]
                          [--verbosity {low,medium,high}]
                          [--reasoning_effort {low,medium,high,xhigh}]
                          [--thinking_level {minimal,low,medium,high}]
                          [--effort {low,medium,high,max}]
                          [--strict_control_acceptance] [--provider PROVIDER]
                          [--system_prompt SYSTEM_PROMPT | --system_prompt_b64 SYSTEM_PROMPT_B64]
                          [--few_shot_examples FEW_SHOT_EXAMPLES]
                          [--prompt_layout {standard,compact}]
                          [--cache_pad_target_tokens CACHE_PAD_TARGET_TOKENS]
                          [--prompt_cache_key PROMPT_CACHE_KEY]
                          [--gemini_cached_content GEMINI_CACHED_CONTENT]
                          [--requesty_auto_cache | --no-requesty_auto_cache]
                          [--vertex_auto_adc_login | --no-vertex_auto_adc_login]
                          [--vertex_access_token_refresh_seconds VERTEX_ACCESS_TOKEN_REFRESH_SECONDS]
                          [--create_gemini_cache]
                          [--gemini_cache_ttl GEMINI_CACHE_TTL]
                          [--gemini_cache_ttl_autoupdate | --no-gemini_cache_ttl_autoupdate]
                          [--keep_gemini_cache] [--enable_cot]
                          [--no_explanation] [--logprobs | --no-logprobs]
                          [--calibration]
                          [--confusion_heatmap | --no-confusion_heatmap]
                          [--api_key_var API_KEY_VAR]
                          [--api_base_var API_BASE_VAR]
                          [--max_retries MAX_RETRIES]
                          [--retry_delay RETRY_DELAY]
                          [--request_interval_ms REQUEST_INTERVAL_MS]
                          [--request_timeout_seconds REQUEST_TIMEOUT_SECONDS]
                          [--threads THREADS]
                          [--prompt_log_detail {full,compact}]
                          [--flush_rows FLUSH_ROWS]
                          [--flush_seconds FLUSH_SECONDS]
                          [--validator_cmd VALIDATOR_CMD]
                          [--validator_args VALIDATOR_ARGS]
                          [--validator_timeout VALIDATOR_TIMEOUT]
                          [--validator_prompt_max_candidates VALIDATOR_PROMPT_MAX_CANDIDATES]
                          [--validator_prompt_max_chars VALIDATOR_PROMPT_MAX_CHARS]
                          [--validator_exhausted_policy {accept_blank_confidence,unclassified,error}]
                          [--validator_debug] [--log_level LOG_LEVEL]
                          [--update-models] [--models-output MODELS_OUTPUT]
                          [--models-providers MODELS_PROVIDERS [MODELS_PROVIDERS ...]]
                          [--summarize-log-errors SUMMARIZE_LOG_ERRORS]
                          [--summarize-log-errors-top SUMMARIZE_LOG_ERRORS_TOP]
                          [--timeout_probe]
                          [--timeout_probe_size TIMEOUT_PROBE_SIZE]
                          [--timeout_probe_repeats TIMEOUT_PROBE_REPEATS]
                          [--timeout_probe_output_dir TIMEOUT_PROBE_OUTPUT_DIR]
                          [--timeout_probe_relaxed | --no-timeout_probe_relaxed]
                          [--metrics_only]

Benchmark an OpenAI model on a linguistic classification dataset.

options:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...]
                        Path(s) to input CSV file(s) with examples (default
                        location: D:\OneDrive - Filozofická fakulta,
                        Univerzita
                        Karlova\Skola\ai\dhAIBench\dhAIBench\data\input).
  --labels LABELS       Optional path to CSV file that provides ground-truth
                        labels (ID;truth).
  --output OUTPUT       Optional output CSV path or directory. When omitted,
                        defaults to
                        <input>__<provider>__<model>__<timestamp>.csv
                        alongside each input file.
  --resume              Resume a run from an existing --output CSV. The agent
                        reconstructs the prior run configuration from the
                        corresponding prompt log and/or metrics artifact when
                        available; any explicitly provided CLI flags take
                        precedence.
  --task_name TASK_NAME
                        Optional task name stored in metrics metadata. If
                        omitted, it is inferred from the output filename.
  --task_description TASK_DESCRIPTION
                        Optional free-text task description stored in metrics
                        metadata.
  --tags TAGS           Optional metrics tags (semicolon-delimited string
                        recommended), stored in metrics metadata.
  --unclassified, --unlcassified
                        Resume helper: with --resume, remove rows whose
                        prediction is 'unclassified' from the existing output
                        and re-prompt only those IDs. Alias --unlcassified is
                        accepted for compatibility.
  --repeat_unclassified, --repeat-unclassified
                        After each pass, automatically rerun the remaining
                        'unclassified' rows in --resume --unclassified mode
                        until none remain, the same IDs remain unclassified
                        twice in a row, or every remaining unclassified row
                        also has truth='unclassified'.
  --model MODEL         Model name (e.g., gpt-4-turbo).
  --temperature TEMPERATURE
                        Sampling temperature. Omit to let the provider/model
                        use its default.
  --top_p TOP_P         Nucleus sampling parameter. Omit to let the
                        provider/model use its default.
  --top_k TOP_K         Top-k sampling (ignored for APIs that do not support
                        it).
  --service_tier {standard,flex,priority}
                        Optional service-tier hint for providers that support
                        differentiated throughput.
  --verbosity {low,medium,high}
                        Optional output verbosity control for GPT models. Sent
                        as verbosity (Chat Completions) or text.verbosity
                        (Responses API).
  --reasoning_effort {low,medium,high,xhigh}
                        Optional reasoning effort level. Sent as
                        reasoning.effort for OpenAI-style models and as
                        reasoning_effort for Gemini targets.
  --thinking_level {minimal,low,medium,high}
                        Optional Gemini thinking level (minimal applies to
                        Gemini Flash models). Sent via
                        extra_body.google.thinking_config for Gemini OpenAI-
                        compatible targets.
  --effort {low,medium,high,max}
                        Optional Claude effort level. Sent as effort when
                        provided.
  --strict_control_acceptance
                        Fail an example when requested controls are rejected
                        or not present in the final successful request
                        payload.
  --provider PROVIDER   Model provider identifier used to look up default
                        credentials. Known providers are preconfigured; custom
                        providers are inferred from <PROVIDER>_API_KEY (or
                        <PROVIDER>_ACCESS_TOKEN) and <PROVIDER>_BASE_URL.
  --system_prompt SYSTEM_PROMPT
                        System prompt injected into the chat completion.
  --system_prompt_b64 SYSTEM_PROMPT_B64
                        Base64-encoded system prompt (used by the GUI to
                        ensure cross-platform commands).
  --few_shot_examples FEW_SHOT_EXAMPLES
                        Number of labeled examples to prepend as few-shot
                        demonstrations.
  --prompt_layout {standard,compact}
                        Prompt payload layout. standard preserves the current
                        verbose payload; compact removes duplicated fields to
                        improve cache reuse.
  --cache_pad_target_tokens CACHE_PAD_TARGET_TOKENS
                        Optional shared-prefix token target for cache padding.
                        If >0, shared-prefix length is calibrated from early
                        prompt structure; subsequent prompts are padded toward
                        this shared-prefix target.
  --prompt_cache_key PROMPT_CACHE_KEY
                        Optional provider cache-routing key (when supported)
                        to improve prompt-cache hit consistency for stable
                        prompt prefixes.
  --gemini_cached_content GEMINI_CACHED_CONTENT
                        Optional Gemini context-cache resource name for
                        providers that expose Gemini OpenAI-compatible caching
                        via extra_body.extra_body.google.cached_content.
                        Mutually exclusive with --create_gemini_cache.
  --requesty_auto_cache, --no-requesty_auto_cache
                        Enable/disable Requesty automatic caching by sending
                        extra_body.requesty.auto_cache. Only used when
                        --provider requesty.
  --vertex_auto_adc_login, --no-vertex_auto_adc_login
                        Enable/disable automatic one-time ADC login for Vertex
                        when credentials are missing (browser-based gcloud
                        auth flow). Only used when --provider vertex.
  --vertex_access_token_refresh_seconds VERTEX_ACCESS_TOKEN_REFRESH_SECONDS
                        Override Vertex access-token refresh interval in
                        seconds. Only used when --provider vertex.
  --create_gemini_cache
                        Auto-create a Gemini CachedContent resource from the
                        system prompt before the benchmark run and delete it
                        afterward (unless --keep_gemini_cache is set). Sets
                        --gemini_cached_content automatically. Mutually
                        exclusive with --gemini_cached_content.
  --gemini_cache_ttl GEMINI_CACHE_TTL
                        Time-to-live in seconds for the auto-created Gemini
                        cache (default: 3600 = 1 hour). Only used when
                        --create_gemini_cache is set.
  --gemini_cache_ttl_autoupdate, --no-gemini_cache_ttl_autoupdate
                        When enabled, automatically refreshes TTL for auto-
                        created Gemini cache during long runs to avoid
                        expiration (target: refresh one hour before expiry).
                        Only used when --create_gemini_cache is set. (default:
                        True)
  --keep_gemini_cache   Do not delete the auto-created Gemini cache after the
                        run. The cache resource name is logged so it can be
                        reused via --gemini_cached_content. Only meaningful
                        when --create_gemini_cache is set.
  --enable_cot          If set, encourages the model to reason step-by-step
                        before answering.
  --no_explanation      Skip requesting explanations to reduce token usage.
  --logprobs, --no-logprobs
                        Enable token log probabilities when supported.
                        Disabled by default for better large-run throughput.
                        Use --logprobs to enable. (default: False)
  --calibration         Generate a calibration plot using the model's
                        confidences.
  --confusion_heatmap, --no-confusion_heatmap
                        Generate a confusion heatmap when label-based metrics
                        are available. Enabled by default; use --no-
                        confusion_heatmap to disable it. (default: True)
  --api_key_var API_KEY_VAR
                        Environment variable name that stores the API key or
                        access token.
  --api_base_var API_BASE_VAR
                        Environment variable name that stores the API base
                        URL.
  --max_retries MAX_RETRIES
                        Maximum number of retry attempts per example on API
                        errors. Vertex/Gemini RESOURCE_EXHAUSTED (HTTP 429)
                        uses a fixed backoff ladder of 5/15/30/60/120 seconds
                        before the run stops.
  --retry_delay RETRY_DELAY
                        Delay (seconds) between API retries.
  --request_interval_ms REQUEST_INTERVAL_MS
                        Minimum delay in milliseconds between outgoing API
                        requests. Use 0 to disable request pacing.
  --request_timeout_seconds REQUEST_TIMEOUT_SECONDS
                        Per-request timeout in seconds for provider API calls.
                        Use 0 or a negative value to disable timeout.
  --threads THREADS     Number of concurrent worker threads used to classify
                        examples. Use 1 to keep sequential processing.
  --prompt_log_detail {full,compact}
                        Prompt-log detail level. full stores full
                        request/response text; compact omits heavy text
                        fields.
  --flush_rows FLUSH_ROWS
                        Flush CSV and NDJSON prompt log after this many
                        committed rows (default: 100).
  --flush_seconds FLUSH_SECONDS
                        Flush CSV and NDJSON prompt log after this many
                        seconds even if flush_rows was not reached (default:
                        2.0).
  --validator_cmd VALIDATOR_CMD
                        Optional path to an NDJSON validator
                        executable/script. When provided, the agent will
                        validate each prediction and may retry with extra
                        constraints. If the path ends with .py it will be run
                        via the current Python interpreter.
  --validator_args VALIDATOR_ARGS
                        Optional extra arguments passed to the validator
                        executable/script as a single string (supports
                        quoting). Example: "--lexicon data/lemmas.txt
                        --max_distance 2 --max_suggestions 30". For the
                        bundled lemmatization validators, validator-side
                        --max_distance 0 disables the distance threshold.
                        Validator-side --max_suggestions caps how many labels
                        the validator returns;
                        --validator_prompt_max_candidates caps how many of
                        those returned labels are rendered into the retry
                        prompt.
  --validator_timeout VALIDATOR_TIMEOUT
                        Timeout (seconds) for each validator request/response
                        roundtrip.
  --validator_prompt_max_candidates VALIDATOR_PROMPT_MAX_CANDIDATES
                        Maximum number of allowed_labels candidates rendered
                        into a validator retry prompt. This is a benchmark-
                        side cap applied after any validator-side limit such
                        as --max_suggestions.
  --validator_prompt_max_chars VALIDATOR_PROMPT_MAX_CHARS
                        Maximum character length of the validator retry
                        instruction appended to the prompt.
  --validator_exhausted_policy {accept_blank_confidence,unclassified,error}
                        What to do when the validator keeps requesting retry
                        but --max_retries is exhausted.
                        accept_blank_confidence keeps the last label but
                        blanks confidence; unclassified forces label to
                        "unclassified"; error aborts the run.
  --validator_debug     Log validator NDJSON send/receive payloads at DEBUG
                        level.
  --log_level LOG_LEVEL
                        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
  --update-models, -updatemodels
                        If set, fetch available models for configured
                        providers and update config_models.js.
  --models-output MODELS_OUTPUT
                        Output path for generated model catalog JS when
                        --update-models is used.
  --models-providers MODELS_PROVIDERS [MODELS_PROVIDERS ...]
                        Optional list of provider slugs to update when
                        --update-models is specified. Custom slugs are
                        allowed; env vars are inferred as <SLUG>_API_KEY (or
                        <SLUG>_ACCESS_TOKEN) and <SLUG>_BASE_URL.
  --summarize-log-errors SUMMARIZE_LOG_ERRORS
                        Path to a prompt log file (*.log) produced by this
                        benchmark. Supports NDJSON and legacy JSON-array logs.
                        Prints a compact attempt/error summary and exits.
  --summarize-log-errors-top SUMMARIZE_LOG_ERRORS_TOP
                        Maximum number of per-example error rows shown by
                        --summarize-log-errors.
  --timeout_probe       Run an automated timeout diagnosis matrix on a fixed
                        subset of the input file and exit. This mode runs
                        multiple short benchmark passes with different
                        timeout/concurrency profiles and prints a compact
                        diagnosis.
  --timeout_probe_size TIMEOUT_PROBE_SIZE
                        Number of input rows sampled for --timeout_probe
                        (default: 60).
  --timeout_probe_repeats TIMEOUT_PROBE_REPEATS
                        Repetitions per timeout-probe scenario (default: 2).
  --timeout_probe_output_dir TIMEOUT_PROBE_OUTPUT_DIR
                        Directory for timeout-probe subset/output/log
                        artifacts.
  --timeout_probe_relaxed, --no-timeout_probe_relaxed
                        Include a relaxed-timeout scenario in --timeout_probe
                        runs (default: enabled). (default: True)
  --metrics_only, --metrics-only
                        Skip model/API calls and compute metrics from existing
                        output CSV file(s) provided via --input. Truth labels
                        are taken from each output truth column and optionally
                        overridden by --labels. When --input is omitted, only
                        agreement_summary.json and agreement_clusters.json are
                        refreshed.
```
