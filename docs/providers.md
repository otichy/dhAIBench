# Providers And Authentication

## Default Behavior

`--provider` selects the credential namespace and provider-specific request behavior. If omitted, the default provider is `openai`.

Credential lookup order is:

1. Explicit CLI overrides via `--api_key_var` and `--api_base_var`
2. Provider-aware defaults inferred from the provider slug
3. `.env`
4. Existing shell environment

Examples:

- `OPENAI_API_KEY`
- `REQUESTY_API_KEY`
- `REQUESTY_BASE_URL`
- `CUSTOMROUTER_ACCESS_TOKEN`
- `CUSTOMROUTER_BASE_URL`

## Request Controls

Use only the controls supported by your target provider:

- `--reasoning_effort {low,medium,high,xhigh}` for OpenAI-style reasoning
- `--service_tier {standard,flex,priority}` for providers with throughput tiers
  (OpenAI, OpenRouter, and Gemini/Vertex expose all three; Claude uses
  `standard` or `priority`). OpenRouter receives the selected tier as the
  top-level `service_tier` request field; support depends on the selected model.
- `--verbosity {low,medium,high}` for GPT output detail
- `--thinking_level {minimal,low,medium,high}` for Gemini thinking configuration
- `--effort {low,medium,high,max}` for Claude-style effort
- `--prompt_cache_key` for OpenAI-style cache routing
- `--openai_cache_breakpoint` for the OpenAI-compatible explicit system-prefix boundary
- `--cache_warmup_delay_seconds` for multithreaded cache propagation
- `--gemini_cached_content` or `--create_gemini_cache` for Gemini context caching
- `--requesty_auto_cache` for Requesty automatic caching
- `--strict_control_acceptance` to fail when requested controls are dropped

For prompt-caching experiments, start with `--prompt_layout compact`.

## OpenRouter Service Tiers

OpenRouter service tiers use the normal model ID plus the top-level
`service_tier` field. For example:

```bash
python benchmark_agent.py --provider openrouter --model openai/gpt-5 \
  --service_tier flex --input data/input/example.csv
```

`flex` is available only on selected models. OpenRouter restricts a flex request
to flex endpoints when they exist; if the model has no flex endpoints, it may
route at the standard tier instead. Check the response metadata and OpenRouter's
model-endpoints listing when validating billing or routing.

## OpenRouter Prompt Caching

For models with automatic prompt caching, keep the shared prefix stable and use:

```bash
python benchmark_agent.py --provider openrouter --model openai/gpt-5 \
  --prompt_layout compact --prompt_cache_key benchmark-prefix-v1 \
  --input data/input/example.csv
```

The cache key is sent as the top-level `prompt_cache_key` field. When no
`session_id` is present, OpenRouter uses it as the sticky-routing key so repeated
requests stay on the same provider endpoint. It does not itself create an
explicit cache breakpoint.

For supporting OpenAI models that expose explicit prompt caching (for example,
GPT-5.6+), add a boundary after the static system/developer prompt:

```bash
python benchmark_agent.py --provider openrouter --model openai/gpt-5.6 \
  --prompt_layout compact --prompt_cache_key benchmark-prefix-v1 \
  --openai_cache_breakpoint --threads 10 --cache_warmup_delay_seconds 5 \
  --input data/input/example.csv
```

`--openai_cache_breakpoint` adds `prompt_cache_breakpoint: {"mode":"explicit"}`
to the last text block of the system/developer message and sends
`prompt_cache_options: {"mode":"explicit","ttl":"30m"}`. This keeps the
row-specific user message outside the explicit cache boundary.

When caching is enabled with multiple threads, the first work item is sent
synchronously. If its usage metadata reports cache-write/create tokens, the
runner waits `--cache_warmup_delay_seconds` (5 seconds by default) before starting
the remaining threads. Set the delay to `0` to disable this barrier. No delay is
applied when the first response reports no cache write.

Cache reads and writes are preserved from
`usage.prompt_tokens_details.cached_tokens` and `cache_write_tokens`. OpenRouter's
top-level `cache_discount` is also retained in each prompt-log response's
`usage_metadata`.

Models with other provider-specific `cache_control` formats still need their own
cache controls; `--openai_cache_breakpoint` is specifically for the
OpenAI-compatible explicit breakpoint format.

## System Prompt Encoding

Use `--system_prompt` for normal single-line prompts. For multiline or shell-sensitive prompts, use `--system_prompt_b64`.

The browser GUI switches automatically:

- one-line prompt -> `--system_prompt`
- multiline prompt -> `--system_prompt_b64`

## Vertex

For `--provider vertex`, authentication is OAuth access-token based rather than static API-key based.

Relevant environment variables:

- `VERTEX_ACCESS_TOKEN_COMMAND`
- `VERTEX_ACCESS_TOKEN`
- `VERTEX_BASE_URL`
- `VERTEX_MODELS_BASE_URL`
- `VERTEX_ACCESS_TOKEN_REFRESH_SECONDS`
- `VERTEX_AUTO_ADC_LOGIN`
- `VERTEX_ADC_LOGIN_COMMAND`

### Browser-Capable Workstation

Recommended setup:

```bash
VERTEX_BASE_URL=https://us-central1-aiplatform.googleapis.com/v1/projects/<PROJECT_ID>/locations/us-central1/endpoints/openapi
VERTEX_AUTO_ADC_LOGIN=true
VERTEX_ACCESS_TOKEN_COMMAND=gcloud auth application-default print-access-token
```

- If ADC is missing, the agent can trigger `gcloud auth application-default login`.
- If Vertex reports a quota-project 403, set it once with `gcloud auth application-default set-quota-project <PROJECT_ID>`.
- The same auth flow is used for normal benchmark runs and `--update-models`.

### CLI-Only Or Remote Environments

Disable browser auto-login:

```bash
VERTEX_AUTO_ADC_LOGIN=false
```

Then use one of:

- a non-interactive `VERTEX_ACCESS_TOKEN_COMMAND`
- pre-provisioned ADC on the machine or runtime identity
- a manually provided short-lived `VERTEX_ACCESS_TOKEN`

CLI equivalents:

- `--no-vertex_auto_adc_login`
- `--vertex_access_token_refresh_seconds N`

### Model Catalog Fallback

When `--update-models` is used, the agent first tries `<MODELS_BASE_URL>/models`, where `<MODELS_BASE_URL>` is:

- `VERTEX_MODELS_BASE_URL` when set
- otherwise `VERTEX_BASE_URL`

If `/models` returns 404, the agent falls back to:

`https://aiplatform.googleapis.com/v1beta1/publishers/google/models`

This allows separate runtime and catalog endpoints.

## Updating `config_models.js`

Regenerate the model catalog consumed by the GUI:

```bash
python benchmark_agent.py --update-models
```

Useful flags:

- `--models-providers` to limit the provider list
- `--models-output` to change the output path

## Custom Providers

Custom OpenAI-compatible providers do not need hardcoded support as long as you provide:

- a provider slug via `--provider`
- credentials in `<PROVIDER>_API_KEY` or `<PROVIDER>_ACCESS_TOKEN`
- a base URL in `<PROVIDER>_BASE_URL`

Example:

```bash
python benchmark_agent.py \
  --provider myrouter \
  --model some-model
```

with environment variables:

```bash
MYROUTER_API_KEY=...
MYROUTER_BASE_URL=https://example.com/v1
```
