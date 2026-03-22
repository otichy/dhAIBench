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
- `--verbosity {low,medium,high}` for GPT output detail
- `--thinking_level {minimal,low,medium,high}` for Gemini thinking configuration
- `--effort {low,medium,high,max}` for Claude-style effort
- `--prompt_cache_key` for OpenAI-style cache routing
- `--gemini_cached_content` or `--create_gemini_cache` for Gemini context caching
- `--requesty_auto_cache` for Requesty automatic caching
- `--strict_control_acceptance` to fail when requested controls are dropped

For prompt-caching experiments, start with `--prompt_layout compact`.

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
