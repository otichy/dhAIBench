(function () {
  "use strict";

  const CLASSIC_VARIANT = "classic";
  const MODE_FIRST_VARIANT = "mode-first";
  const CLASSIC_STORAGE_KEY = "benchmarkAgentConfig.v1";
  const MODE_FIRST_STORAGE_KEY = "benchmarkAgentModeFirst.v1";
  const MODE_FIRST_UI_STORAGE_KEY = "benchmarkAgentModeFirst.ui.v1";
  const DEFAULT_CLASSIC_COMMAND = 'python benchmark_agent.py --input "" --model ""';
  const PREVIEW_MODES = ["run", "resume", "metrics", "validator"];
  const MODE_LABELS = {
    run: "Run",
    resume: "Resume",
    metrics: "Metrics only",
    validator: "Run & Validate",
  };

  const PROMPT_ESTIMATE_SAMPLE_LIMIT = 100;
  const PROMPT_CACHE_HINT_TOKENS = 1024;
  const MAX_CACHE_PADDING_TOKENS = 200000;
  const NODE_MARKER_LEFT = "\u27E6";
  const NODE_MARKER_RIGHT = "\u27E7";
  const LEGACY_NODE_MARKER_LEFT = "\u00E2\u017A\u00A6";
  const LEGACY_NODE_MARKER_RIGHT = "\u00E2\u017A\u00A7";
  const NODE_MARKER_VARIANTS = [
    [NODE_MARKER_LEFT, NODE_MARKER_RIGHT],
    [LEGACY_NODE_MARKER_LEFT, LEGACY_NODE_MARKER_RIGHT],
  ];
  const MANDATORY_SYSTEM_APPEND =
    `Classify ONLY the text that is explicitly wrapped inside ${NODE_MARKER_LEFT} ${NODE_MARKER_RIGHT} ` +
    "(the 'node' or its marked sub-span). " +
    "Use the surrounding context as supporting evidence, but never change the focus away from the highlighted text. " +
    'If you cannot determine the class/label for the node, return "unclassified".';
  const PROMPT_PREVIEW_PLACEHOLDER_ROW = Object.freeze({
    leftContext: "[leftContext from first sampled CSV row]",
    node: "[node from first sampled CSV row]",
    rightContext: "[rightContext from first sampled CSV row]",
    info: "[info from first sampled CSV row]",
  });
  const CACHE_PADDING_PREFIX =
    "Cache-normalization filler block. Ignore this block for classification.\nCACHE_PAD_BEGIN";
  const CACHE_PADDING_TOKEN = " cachepad";
  const CACHE_PADDING_SUFFIX = "\nCACHE_PAD_END";

  const defaultValues = {
    provider: "openai",
    temperature: "",
    top_p: "",
    verbosity: "",
    reasoning_effort: "",
    thinking_level: "",
    effort: "",
    logprobs: false,
    strict_control_acceptance: true,
    request_interval_ms: "0",
    threads: "1",
    prompt_log_detail: "full",
    flush_rows: "100",
    flush_seconds: "2.0",
    request_timeout_seconds: "30.0",
    max_retries: "3",
    retry_delay: "5.0",
    few_shot_examples: "0",
    prompt_layout: "standard",
    cache_pad_target_tokens: "0",
    prompt_cache_key: "",
    requesty_auto_cache: false,
    vertex_auto_adc_login: true,
    vertex_access_token_refresh_seconds: "3300",
    gemini_cached_content: "",
    create_gemini_cache: false,
    gemini_cache_ttl: "3600",
    gemini_cache_ttl_autoupdate: true,
    keep_gemini_cache: false,
    service_tier: "standard",
    include_explanations: true,
    enable_cot: true,
    api_key_var: "",
    api_base_var: "",
    system_prompt:
      "You are a linguistic classifier that excels at semantic disambiguation.",
    validator_enable: false,
    validator_cmd: "",
    validator_args: "",
    validator_lexicon: "",
    validator_max_distance: "",
    validator_max_distance_per_retry: "",
    validator_max_suggestions: "30",
    validator_timeout: "5.0",
    validator_prompt_max_candidates: "50",
    validator_prompt_max_chars: "8000",
    validator_exhausted_policy: "accept_blank_confidence",
    validator_debug: false,
    metrics_only: false,
    reprompt_unclassified: false,
    repeat_unclassified: false,
    calibration: true,
    confusion_heatmap: true,
    task_name: "",
    task_description: "",
    tags: "",
  };

  const providerDefaults = {
    openai: { apiKeyVar: "OPENAI_API_KEY", apiBaseVar: "OPENAI_BASE_URL" },
    anthropic: {
      apiKeyVar: "ANTHROPIC_API_KEY",
      apiBaseVar: "ANTHROPIC_BASE_URL",
    },
    cohere: { apiKeyVar: "COHERE_API_KEY", apiBaseVar: "COHERE_BASE_URL" },
    google: { apiKeyVar: "GOOGLE_API_KEY", apiBaseVar: "GOOGLE_BASE_URL" },
    vertex: {
      apiKeyVar: "VERTEX_ACCESS_TOKEN",
      apiBaseVar: "VERTEX_BASE_URL",
    },
    huggingface: {
      apiKeyVar: "HF_API_KEY",
      apiBaseVar: "HF_BASE_URL",
    },
    "e-infra": {
      apiKeyVar: "E-INFRA_API_KEY",
      apiBaseVar: "E-INFRA_BASE_URL",
    },
    requesty: {
      apiKeyVar: "REQUESTY_API_KEY",
      apiBaseVar: "REQUESTY_BASE_URL",
    },
  };

  const knownProviderLabels = {
    openai: "OpenAI-Compatible",
    anthropic: "Anthropic (OpenAI proxy)",
    cohere: "Cohere (OpenAI proxy)",
    google: "Google (OpenAI proxy)",
    vertex: "Vertex AI (OpenAI proxy)",
    huggingface: "Hugging Face (OpenAI proxy)",
    "e-infra": "E-INFRA (OpenAI proxy)",
    requesty: "Requesty (OpenAI proxy)",
  };

  const inputHelpTextById = {
    input_path: "One or more CSV input files. Use one path per line.",
    model: "Model identifier. Click and pick from cached models or type your own model id.",
    provider: "Provider slug used by --provider. Refresh updates this list from config_models.js.",
    system_prompt: "Optional system prompt. Multi-line prompts are encoded as --system_prompt_b64.",
    enable_cot: "Encourage model to 'Think ... step-by-step' to elicit chain-of-thought behaviour.",
    include_explanations: "If unchecked, adds --no_explanation to request compact model outputs.",
    logprobs: "If checked, adds --logprobs to request token-level probability estimates (off by default).",
    temperature: "Optional temperature sampling value. Leave blank to omit this parameter.",
    top_p: "Optional nucleus sampling parameter. Leave blank to omit this parameter.",
    top_k: "Optional top-k sampling parameter, if your provider supports it.",
    request_interval_ms: "Minimum delay in milliseconds between API requests.",
    threads: "Number of concurrent worker threads. Output rows are still written in input order.",
    prompt_log_detail: "Prompt-log payload level. compact omits heavy request/response text fields.",
    flush_rows: "Flush CSV and prompt log after this many committed rows.",
    flush_seconds: "Flush CSV and prompt log after this many seconds even when flush_rows is not reached.",
    request_timeout_seconds:
      "Per-request API timeout in seconds. Set 0 to disable timeouts (requests may block for a long time).",
    service_tier: "OpenAI-compatible service tier. Ignored by providers that do not support it.",
    reasoning_effort:
      "Reasoning effort level (low, medium, high, xhigh). Maps to OpenAI reasoning.effort or Gemini reasoning_effort.",
    verbosity:
      "GPT output verbosity level (low, medium, high). Maps to verbosity in Chat Completions and text.verbosity in Responses API.",
    thinking_level:
      "Gemini thinking level (minimal, low, medium, high). Minimal applies only to Gemini Flash models. For Gemini targets this maps to extra_body.google.thinking_config.",
    effort: "Claude effort control (low, medium, high, max). Adds --effort.",
    strict_control_acceptance:
      "Abort examples where requested controls are rejected or omitted from the final successful request.",
    few_shot_examples: "Number of few-shot examples pulled from labels to include in prompt.",
    prompt_layout:
      "Prompt payload shape. compact reduces duplicated per-example text to improve cache reuse.",
    cache_pad_target_tokens:
      "Optional shared-prefix cache padding target. Runtime pads only the cacheable shared prefix and does not include row-specific payload fields in this estimate.",
    prompt_cache_key:
      "Optional routing key sent as --prompt_cache_key for providers that support prompt cache keying.",
    requesty_auto_cache:
      "Enable Requesty auto caching by adding --requesty_auto_cache (maps to extra_body.requesty.auto_cache).",
    vertex_auto_adc_login:
      "If enabled for Vertex, automatically launches `gcloud auth application-default login` once when ADC is missing.",
    vertex_access_token_refresh_seconds:
      "Vertex access-token refresh interval in seconds (default: 3300).",
    gemini_cached_content:
      "Optional Gemini context-cache resource name sent as --gemini_cached_content (Gemini OpenAI-compatible endpoints).",
    create_gemini_cache:
      "Automatically create a Gemini CachedContent from the system prompt before the run and delete it afterward.",
    gemini_cache_ttl:
      "Time-to-live in seconds for the auto-created Gemini cache (default: 3600 = 1 hour).",
    gemini_cache_ttl_autoupdate:
      "When enabled, auto-refreshes cache TTL during long runs to avoid expiration.",
    keep_gemini_cache:
      "Do not delete the auto-created cache after the run. The resource name is logged for reuse.",
    labels_path: "Optional labels CSV path. If omitted, labels are expected in the input file.",
    task_name:
      "Optional task name written to metrics metadata. Leave blank to infer from output filename.",
    task_description: "Optional free-text task description written to metrics metadata.",
    tags:
      "Optional metrics tags. Use semicolon-separated tags (for example: tag1;tag2;tag3).",
    metrics_only:
      "Skip API calls and only recompute metrics from existing output CSVs listed in input_path. Truth labels come from output truth column and can be overridden via labels_path.",
    output_path:
      "Optional output CSV file path, or output directory when running multiple inputs. Resume mode requires an existing output CSV here.",
    reprompt_unclassified:
      "When checked, emits --unclassified so --resume re-prompts only rows currently labeled unclassified.",
    repeat_unclassified:
      "When checked, emits --repeat_unclassified so the run keeps retrying remaining unclassified rows until they are resolved, stabilize across two passes, or all match truth='unclassified'.",
    api_key_var: "Environment variable that stores the API key or access token used for this run.",
    api_base_var: "Environment variable containing the provider base URL.",
    calibration: "Generate a calibration plot from model confidence scores.",
    confusion_heatmap: "Generate a confusion heatmap when label-based metrics are available.",
    validator_enable: "Enable external validator roundtrip and retry logic.",
    validator_cmd: "Path to the validator executable or .py script used for label validation.",
    validator_args:
      "The GUI synthesizes --validator_args from the dedicated validator lexicon, max distance, distance increment per retry, and max suggestions fields.",
    validator_lexicon:
      "Optional value passed to the validator as --lexicon. Leave blank to let the validator script use its own default lexicon.",
    validator_max_distance:
      "Optional value passed to the validator as --max_distance. For the bundled lemmatization validators, 0 disables the distance threshold. Returned labels are still capped by the lexicon and --max_suggestions.",
    validator_max_distance_per_retry:
      "Optional value passed to the validator as --max_distance_per_retry. The increment starts only on the second retry, meaning the third overall attempt is the first one with a higher threshold. Leave blank or 0 to keep the validator threshold fixed.",
    validator_max_suggestions:
      "Optional value passed to the validator as --max_suggestions. It caps how many labels the validator returns after any distance filtering, before the benchmark-side Max prompt candidates cap is applied.",
    validator_timeout: "Timeout in seconds for each validator invocation.",
    validator_prompt_max_candidates:
      "Benchmark-side cap for how many validator-returned labels are shown in the retry prompt. This can be lower than the validator's own --max_suggestions limit.",
    validator_prompt_max_chars: "Hard cap for validator retry prompt character length.",
    validator_exhausted_policy: "Action when validator retries are exhausted.",
    validator_debug: "Emit verbose validator payload logging (DEBUG mode).",
    max_retries:
      "Maximum number of retry attempts per example on API errors and validator-driven retries.",
    retry_delay:
      "Delay in seconds between retry attempts after API errors or validator-driven retries.",
  };

  const cliFlagReferenceSections = [
    {
      title: "Setup & Modes",
      entries: [
        {
          flags: ["--input"],
          helpId: "input_path",
          modes: ["Run", "Run & Validate", "Metrics only"],
        },
        {
          flags: ["--model"],
          helpId: "model",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--output"],
          helpId: "output_path",
          modes: ["Run", "Run & Validate", "Resume"],
        },
        {
          flags: ["--resume"],
          description:
            "Resume an existing output CSV and recover prior run settings from prompt-log and metrics artifacts when available.",
          modes: ["Resume"],
        },
        {
          flags: ["--metrics_only"],
          helpId: "metrics_only",
          modes: ["Metrics only"],
        },
        {
          flags: ["--labels"],
          helpId: "labels_path",
          modes: ["Run", "Run & Validate", "Metrics only"],
        },
        {
          flags: ["--unclassified"],
          helpId: "reprompt_unclassified",
          modes: ["Resume"],
        },
        {
          flags: ["--repeat_unclassified"],
          helpId: "repeat_unclassified",
          modes: ["Run", "Run & Validate", "Resume"],
        },
      ],
    },
    {
      title: "Prompt Strategy",
      entries: [
        {
          flags: ["--system_prompt", "--system_prompt_b64"],
          description:
            "Optional system prompt. Single-line text is emitted as --system_prompt; multi-line text is encoded as --system_prompt_b64.",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--enable_cot"],
          helpId: "enable_cot",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--no_explanation"],
          helpId: "include_explanations",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--few_shot_examples"],
          helpId: "few_shot_examples",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--prompt_layout"],
          helpId: "prompt_layout",
          modes: ["Run", "Run & Validate"],
        },
      ],
    },
    {
      title: "Execution & Logging",
      entries: [
        {
          flags: ["--temperature"],
          helpId: "temperature",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--top_p"],
          helpId: "top_p",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--top_k"],
          helpId: "top_k",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--request_interval_ms"],
          helpId: "request_interval_ms",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--threads"],
          helpId: "threads",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--request_timeout_seconds"],
          helpId: "request_timeout_seconds",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--max_retries"],
          helpId: "max_retries",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--retry_delay"],
          helpId: "retry_delay",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--logprobs"],
          helpId: "logprobs",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--prompt_log_detail"],
          helpId: "prompt_log_detail",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--flush_rows"],
          helpId: "flush_rows",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--flush_seconds"],
          helpId: "flush_seconds",
          modes: ["Run", "Run & Validate"],
        },
      ],
    },
    {
      title: "Provider Controls",
      entries: [
        {
          flags: ["--provider"],
          helpId: "provider",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--service_tier"],
          helpId: "service_tier",
          modes: ["Run", "Run & Validate"],
          providers: ["OpenAI"],
        },
        {
          flags: ["--reasoning_effort"],
          helpId: "reasoning_effort",
          modes: ["Run", "Run & Validate"],
          providers: ["OpenAI", "Gemini"],
        },
        {
          flags: ["--verbosity"],
          helpId: "verbosity",
          modes: ["Run", "Run & Validate"],
          providers: ["OpenAI GPT"],
        },
        {
          flags: ["--thinking_level"],
          helpId: "thinking_level",
          modes: ["Run", "Run & Validate"],
          providers: ["Gemini"],
        },
        {
          flags: ["--effort"],
          helpId: "effort",
          modes: ["Run", "Run & Validate"],
          providers: ["Claude"],
        },
        {
          flags: ["--strict_control_acceptance"],
          helpId: "strict_control_acceptance",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--cache_pad_target_tokens"],
          helpId: "cache_pad_target_tokens",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--prompt_cache_key"],
          helpId: "prompt_cache_key",
          modes: ["Run", "Run & Validate"],
        },
        {
          flags: ["--requesty_auto_cache"],
          helpId: "requesty_auto_cache",
          modes: ["Run", "Run & Validate"],
          providers: ["Requesty"],
        },
        {
          flags: ["--no-vertex_auto_adc_login"],
          helpId: "vertex_auto_adc_login",
          modes: ["Run", "Run & Validate"],
          providers: ["Vertex"],
          note: "The GUI emits the negative form only when auto-login is turned off.",
        },
        {
          flags: ["--vertex_access_token_refresh_seconds"],
          helpId: "vertex_access_token_refresh_seconds",
          modes: ["Run", "Run & Validate"],
          providers: ["Vertex"],
        },
        {
          flags: ["--gemini_cached_content"],
          helpId: "gemini_cached_content",
          modes: ["Run", "Run & Validate"],
          providers: ["Gemini"],
        },
        {
          flags: ["--create_gemini_cache"],
          helpId: "create_gemini_cache",
          modes: ["Run", "Run & Validate"],
          providers: ["Gemini"],
        },
        {
          flags: ["--gemini_cache_ttl"],
          helpId: "gemini_cache_ttl",
          modes: ["Run", "Run & Validate"],
          providers: ["Gemini"],
        },
        {
          flags: ["--no-gemini_cache_ttl_autoupdate"],
          helpId: "gemini_cache_ttl_autoupdate",
          modes: ["Run", "Run & Validate"],
          providers: ["Gemini"],
          note: "The GUI emits the negative form only when TTL auto-refresh is turned off.",
        },
        {
          flags: ["--keep_gemini_cache"],
          helpId: "keep_gemini_cache",
          modes: ["Run", "Run & Validate"],
          providers: ["Gemini"],
        },
      ],
    },
    {
      title: "Evaluation & Metadata",
      entries: [
        {
          flags: ["--task_name"],
          helpId: "task_name",
          modes: ["Run", "Run & Validate", "Metrics only"],
        },
        {
          flags: ["--task_description"],
          helpId: "task_description",
          modes: ["Run", "Run & Validate", "Metrics only"],
        },
        {
          flags: ["--tags"],
          helpId: "tags",
          modes: ["Run", "Run & Validate", "Metrics only"],
        },
        {
          flags: ["--calibration"],
          helpId: "calibration",
          modes: ["Run", "Run & Validate", "Metrics only"],
        },
        {
          flags: ["--no-confusion_heatmap"],
          helpId: "confusion_heatmap",
          modes: ["Run", "Run & Validate", "Metrics only"],
          note: "The GUI emits the negative form only when heatmap generation is turned off.",
        },
      ],
    },
    {
      title: "Validator",
      entries: [
        {
          flags: ["--validator_cmd"],
          helpId: "validator_cmd",
          modes: ["Run & Validate"],
        },
        {
          flags: ["--validator_args"],
          helpId: "validator_args",
          modes: ["Run & Validate"],
        },
        {
          flags: ["--validator_timeout"],
          helpId: "validator_timeout",
          modes: ["Run & Validate"],
        },
        {
          flags: ["--validator_prompt_max_candidates"],
          helpId: "validator_prompt_max_candidates",
          modes: ["Run & Validate"],
        },
        {
          flags: ["--validator_prompt_max_chars"],
          helpId: "validator_prompt_max_chars",
          modes: ["Run & Validate"],
        },
        {
          flags: ["--validator_exhausted_policy"],
          helpId: "validator_exhausted_policy",
          modes: ["Run & Validate"],
        },
        {
          flags: ["--validator_debug"],
          helpId: "validator_debug",
          modes: ["Run & Validate"],
        },
      ],
    },
  ];
  let fieldHelpIdCounter = 0;

  function getWindowModelCatalog() {
    return window.MODEL_CATALOG && typeof window.MODEL_CATALOG === "object"
      ? window.MODEL_CATALOG
      : {};
  }

  function getWindowPricingCatalog() {
    return window.MODEL_PRICING_CATALOG && typeof window.MODEL_PRICING_CATALOG === "object"
      ? window.MODEL_PRICING_CATALOG
      : {};
  }

  function normalizePricingProviderKey(provider) {
    const normalized = (provider || "").toString().trim().toLowerCase();
    if (normalized === "einfra") {
      return "e-infra";
    }
    return normalized;
  }

  function getSelectedServiceTier(ctx) {
    const raw = ctx.serviceTierInput?.value || defaultValues.service_tier;
    return ["standard", "flex", "priority", "batch"].includes(raw) ? raw : "standard";
  }

  function formatPriceAmount(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return "N/A";
    }
    return `$${value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  }

  function getPricingServiceTiers(entry) {
    return entry && entry.service_tiers && typeof entry.service_tiers === "object"
      ? entry.service_tiers
      : {};
  }

  function pricingEntryHasRef(entry) {
    return !!(entry && typeof entry.pricing_ref === "string" && entry.pricing_ref.trim());
  }

  function pricingEntryHasReason(entry) {
    return !!(entry && typeof entry.reason === "string" && entry.reason.trim());
  }

  function pricingEntryHasUsableRates(entry) {
    return Object.values(getPricingServiceTiers(entry)).some((tierRates) => {
      if (!tierRates || typeof tierRates !== "object") {
        return false;
      }
      return ["input_usd_per_mtokens", "cached_input_usd_per_mtokens", "output_usd_per_mtokens"]
        .some((key) => typeof tierRates[key] === "number" && Number.isFinite(tierRates[key]));
    });
  }

  function classifyPricingCatalogEntry(entry) {
    if (!entry || typeof entry !== "object") {
      return "missing";
    }
    if (pricingEntryHasRef(entry)) {
      return "alias";
    }
    if (pricingEntryHasUsableRates(entry)) {
      return "priced";
    }
    if (entry.needs_manual_update === true) {
      return "unpriced";
    }
    if (pricingEntryHasReason(entry)) {
      return "unsupported";
    }
    return "unpriced";
  }

  function resolvePricingCatalogEntry(ctx, provider, modelId) {
    const catalog = ctx.priceCatalog;
    const normalizedProvider = normalizePricingProviderKey(provider);
    if (
      !catalog ||
      typeof catalog !== "object" ||
      !catalog.providers ||
      typeof catalog.providers !== "object" ||
      !catalog.providers[normalizedProvider] ||
      !catalog.providers[normalizedProvider].models ||
      typeof catalog.providers[normalizedProvider].models !== "object"
    ) {
      return null;
    }
    const providerModels = catalog.providers[normalizedProvider].models;
    let currentKey = modelId;
    let safety = 0;
    while (currentKey && providerModels[currentKey] && safety < 12) {
      const entry = providerModels[currentKey];
      if (!entry || typeof entry !== "object") {
        return null;
      }
      if (!pricingEntryHasRef(entry)) {
        return { entry, resolvedKey: currentKey };
      }
      currentKey = typeof entry.pricing_ref === "string" ? entry.pricing_ref.trim() : "";
      safety += 1;
    }
    return null;
  }

  function buildFlagReferenceDescription(entry) {
    const parts = [];
    const primary =
      typeof entry.description === "string" && entry.description.trim()
        ? entry.description.trim()
        : typeof entry.helpId === "string" && inputHelpTextById[entry.helpId]
          ? inputHelpTextById[entry.helpId].trim()
          : "";
    if (primary) {
      parts.push(primary);
    }
    if (typeof entry.note === "string" && entry.note.trim()) {
      parts.push(entry.note.trim());
    }
    return parts.join(" ");
  }

  function createFlagReferenceChip(text) {
    const chip = document.createElement("span");
    chip.className = "cli-reference-chip";
    chip.textContent = text;
    return chip;
  }

  function renderCliFlagReference(ctx) {
    if (!ctx.cliFlagReference) {
      return;
    }
    ctx.cliFlagReference.innerHTML = "";
    const fragment = document.createDocumentFragment();
    cliFlagReferenceSections.forEach((section) => {
      const group = document.createElement("section");
      group.className = "cli-reference-group";

      const heading = document.createElement("h3");
      heading.textContent = section.title;
      group.appendChild(heading);

      const list = document.createElement("div");
      list.className = "cli-reference-list";

      section.entries.forEach((entry) => {
        const item = document.createElement("article");
        item.className = "cli-reference-item";

        const flags = document.createElement("div");
        flags.className = "cli-reference-flags";
        (entry.flags || []).forEach((flag) => {
          const code = document.createElement("code");
          code.textContent = flag;
          flags.appendChild(code);
        });
        item.appendChild(flags);

        const meta = document.createElement("div");
        meta.className = "cli-reference-meta";
        if (Array.isArray(entry.modes) && entry.modes.length > 0) {
          meta.appendChild(createFlagReferenceChip(`Modes: ${entry.modes.join(" / ")}`));
        }
        if (Array.isArray(entry.providers) && entry.providers.length > 0) {
          meta.appendChild(
            createFlagReferenceChip(`Providers: ${entry.providers.join(" / ")}`)
          );
        }
        if (meta.childNodes.length > 0) {
          item.appendChild(meta);
        }

        const description = document.createElement("p");
        description.textContent = buildFlagReferenceDescription(entry);
        item.appendChild(description);

        list.appendChild(item);
      });

      group.appendChild(list);
      fragment.appendChild(group);
    });
    ctx.cliFlagReference.appendChild(fragment);
  }

  function summarizeModelPricing(ctx, provider, modelId) {
    const resolved = resolvePricingCatalogEntry(ctx, provider, modelId);
    if (!resolved) {
      return "";
    }
    const { entry } = resolved;
    const entryKind = classifyPricingCatalogEntry(entry);
    if (entryKind === "unpriced") {
      return "manual pricing needed";
    }
    if (entryKind === "unsupported") {
      return "no compatible token pricing";
    }
    if (entryKind !== "priced") {
      return "";
    }

    const requestedTier = getSelectedServiceTier(ctx);
    const serviceTiers = getPricingServiceTiers(entry);
    const preferredTier =
      serviceTiers[requestedTier] && typeof serviceTiers[requestedTier] === "object"
        ? serviceTiers[requestedTier]
        : null;
    const standardTier =
      serviceTiers.standard && typeof serviceTiers.standard === "object" ? serviceTiers.standard : null;
    const tier = preferredTier || standardTier;
    if (!tier) {
      return `${requestedTier} unavailable`;
    }
    const tierPrefix = preferredTier ? requestedTier : `${requestedTier} unavailable; standard`;
    return `${tierPrefix} ~ in ${formatPriceAmount(tier.input_usd_per_mtokens)} | cache ${formatPriceAmount(
      tier.cached_input_usd_per_mtokens
    )} | out ${formatPriceAmount(tier.output_usd_per_mtokens)} / 1M`;
  }

  function getProviderCatalogModels(ctx, provider) {
    const entry = ctx.modelCatalog && typeof ctx.modelCatalog === "object" ? ctx.modelCatalog[provider] : null;
    return entry && Array.isArray(entry.models) ? entry.models.filter((value) => typeof value === "string") : [];
  }

  function updateModelCatalogMeta(ctx, provider, models = getProviderCatalogModels(ctx, provider)) {
    if (!ctx.modelCatalogMeta) {
      return;
    }
    const entry = ctx.modelCatalog && typeof ctx.modelCatalog === "object" ? ctx.modelCatalog[provider] : null;
    let note = "";
    if (models.length) {
      note = `Loaded ${models.length} cached model${models.length === 1 ? "" : "s"} for "${provider}".`;
      if (entry && typeof entry.timestamp === "string") {
        note += ` Updated ${entry.timestamp}.`;
      }
      if (entry && entry.error) {
        note += ` Last fetch issue: ${entry.error}.`;
      }
      note += ` Prices shown for ${getSelectedServiceTier(ctx)} tier when available.`;
    } else {
      note = `No cached models for "${provider}". Run python benchmark_agent.py --update-models to refresh.`;
    }

    const selectedModel = ctx.modelInput && typeof ctx.modelInput.value === "string" ? ctx.modelInput.value.trim() : "";
    if (selectedModel) {
      const selectedSummary = summarizeModelPricing(ctx, provider, selectedModel);
      if (selectedSummary) {
        note += ` Selected: ${selectedModel} | ${selectedSummary}.`;
      }
    }

    ctx.modelCatalogMeta.textContent = note;
  }

  function normalizeMode(rawMode) {
    const normalized = (rawMode || "").toString().trim().toLowerCase();
    return PREVIEW_MODES.includes(normalized) ? normalized : "run";
  }

  function isModeFirstVariant(variant) {
    return variant === MODE_FIRST_VARIANT;
  }

  function getConfigStorageKey(variant) {
    if (variant === MODE_FIRST_VARIANT) {
      return MODE_FIRST_STORAGE_KEY;
    }
    return CLASSIC_STORAGE_KEY;
  }

  function getModeFirstUiStorageKey(variant) {
    if (variant === MODE_FIRST_VARIANT) {
      return MODE_FIRST_UI_STORAGE_KEY;
    }
    return null;
  }

  function getModeLabel(mode) {
    return MODE_LABELS[normalizeMode(mode)] || MODE_LABELS.run;
  }

  function providerSlugToEnvPrefix(slug) {
    const safe = (slug || "")
      .toString()
      .trim()
      .toUpperCase()
      .replace(/[^A-Z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
    return safe || "PROVIDER";
  }

  function shellQuote(value) {
    if (value === undefined || value === null) {
      return '""';
    }
    const stringValue = String(value);
    if (!stringValue.length) {
      return '""';
    }
    if (/^[A-Za-z0-9._@%+=:,\\/\\-]+$/.test(stringValue)) {
      return stringValue;
    }
    const escaped = stringValue.replace(/"/g, '""');
    return `"${escaped}"`;
  }

  function buildValidatorArgsValue(data) {
    const parts = [];
    const validatorLexicon = data.get("validator_lexicon")?.toString().trim() ?? "";
    if (validatorLexicon) {
      parts.push("--lexicon", shellQuote(validatorLexicon));
    }
    const validatorMaxDistance = data.get("validator_max_distance")?.toString().trim() ?? "";
    if (validatorMaxDistance) {
      parts.push("--max_distance", validatorMaxDistance);
    }
    const validatorMaxDistancePerRetry =
      data.get("validator_max_distance_per_retry")?.toString().trim() ?? "";
    if (validatorMaxDistancePerRetry) {
      parts.push("--max_distance_per_retry", validatorMaxDistancePerRetry);
    }
    const validatorMaxSuggestions = data.get("validator_max_suggestions")?.toString().trim() ?? "";
    if (
      validatorMaxSuggestions &&
      validatorMaxSuggestions !== defaultValues.validator_max_suggestions
    ) {
      parts.push("--max_suggestions", validatorMaxSuggestions);
    }
    return parts.join(" ").trim();
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function classifyCommandToken(token, index, tokens) {
    if (index === 0) return "cmd-bin";
    if (index === 1) return "cmd-script";
    if (token.startsWith("--")) return "cmd-flag";
    const prev = tokens[index - 1] || "";
    if (prev.startsWith("--")) return "cmd-value";
    return "cmd-plain";
  }

  function renderCommandTokens(tokens) {
    return tokens
      .map((token, index) => {
        const cls = classifyCommandToken(token, index, tokens);
        return `<span class="cmd-token ${cls}">${escapeHtml(token)}</span>`;
      })
      .join('<span class="cmd-space"> </span>');
  }

  function buildUserInstructionLines(layout, enableCot, includeExplanations) {
    const normalizedLayout =
      (layout || defaultValues.prompt_layout || "standard").toString().trim().toLowerCase() === "compact"
        ? "compact"
        : "standard";
    const markerPair = `${NODE_MARKER_LEFT}...${NODE_MARKER_RIGHT}`;
    const userInstructions =
      normalizedLayout === "standard"
        ? [
            `You will receive a text excerpt with separate left/right context fields and a marked example where the node is wrapped as ${NODE_MARKER_LEFT}node${NODE_MARKER_RIGHT}.`,
            `When the node itself contains inner ${markerPair} spans, those marked passages are the classification target; the rest of the node and the contexts remain useful evidence only.`,
            "Identify the label that best matches the required span according to the task definition.",
            "The payload includes a classification_target helper indicating exactly which text must be classified.",
          ]
        : [
            "You will receive left_context, node, and right_context fields for a text excerpt.",
            `If the node contains inner ${markerPair} spans, classify only those marked spans; otherwise classify the full node.`,
            "Identify the label that best matches the required span according to the task definition.",
            "The payload includes a classification_target helper indicating exactly which text must be classified.",
          ];

    if (enableCot) {
      userInstructions.splice(
        2,
        0,
        "Think through the linguistic evidence step-by-step before committing to the label."
      );
    }

    if (includeExplanations) {
      userInstructions.push(
        "Return a JSON object with keys: label (string), explanation (string), confidence (float in [0,1]), node_echo (string), span_source (string)."
      );
    } else {
      userInstructions.push(
        "Return a JSON object with keys: label (string), confidence (float in [0,1]), node_echo (string), span_source (string). Do not include an explanation field."
      );
    }

    userInstructions.push(
      `Set span_source to "node" when the entire node is being classified. If any inner ${markerPair} spans exist, set span_source to "marked_subspan" and set node_echo to exactly the marked text (join multiple marked spans with a single space, in order).`
    );
    userInstructions.push(
      "An additional field named 'info' may provide guidance or metadata relevant to the label; factor it into your decision."
    );
    userInstructions.push(
      "Contract: if node_echo or span_source fail to meet these requirements, the response will be rejected."
    );
    userInstructions.push("Do not include any text outside the JSON object.");
    return userInstructions;
  }

  function getPromptEstimateConfig(data) {
    const rawSystemPrompt = data.get("system_prompt");
    const trimmedSystemPrompt =
      typeof rawSystemPrompt === "string" ? rawSystemPrompt.trim() : "";
    const effectiveSystemPrompt = trimmedSystemPrompt || defaultValues.system_prompt;
    const systemMessage = `${effectiveSystemPrompt}\n\n${MANDATORY_SYSTEM_APPEND}`;

    const layoutRaw = (data.get("prompt_layout") || defaultValues.prompt_layout || "standard")
      .toString()
      .trim()
      .toLowerCase();
    const layout = layoutRaw === "compact" ? "compact" : "standard";
    const includeExplanations = Boolean(data.get("include_explanations"));
    const enableCot = Boolean(data.get("enable_cot"));
    const fewShotRaw = (data.get("few_shot_examples") || "0").toString().trim();
    const parsedFewShot = Number.parseInt(fewShotRaw || "0", 10);
    const fewShotCount =
      Number.isFinite(parsedFewShot) && parsedFewShot > 0 ? parsedFewShot : 0;

    return {
      layout,
      fewShotCount,
      systemMessage,
      instructionsText: buildUserInstructionLines(
        layout,
        enableCot,
        includeExplanations
      ).join("\n"),
    };
  }

  function estimateTokensFromChars(charCount) {
    const safeChars = Math.max(1, charCount);
    return {
      estimateMid: Math.max(1, Math.round(safeChars / 4)),
      estimateLow: Math.max(1, Math.round(safeChars / 4.5)),
      estimateHigh: Math.max(1, Math.round(safeChars / 3.5)),
    };
  }

  function serializePayloadForEstimate(layout, payload) {
    return layout === "standard"
      ? JSON.stringify(payload, null, 2)
      : JSON.stringify(payload, null, 0);
  }

  function extractMarkedSpansForPair(nodeText, markerLeft, markerRight) {
    const node = (nodeText || "").toString();
    const spans = [];
    let searchStart = 0;
    while (searchStart < node.length) {
      const leftIndex = node.indexOf(markerLeft, searchStart);
      if (leftIndex < 0) {
        break;
      }
      const spanStart = leftIndex + markerLeft.length;
      const rightIndex = node.indexOf(markerRight, spanStart);
      if (rightIndex < 0) {
        break;
      }
      const span = node.slice(spanStart, rightIndex).trim();
      if (span) {
        spans.push(span);
      }
      searchStart = rightIndex + NODE_MARKER_RIGHT.length;
    }
    return spans;
  }

  function extractMarkedSpans(nodeText) {
    for (const [markerLeft, markerRight] of NODE_MARKER_VARIANTS) {
      const spans = extractMarkedSpansForPair(nodeText, markerLeft, markerRight);
      if (spans.length > 0) {
        return spans;
      }
    }
    return [];
  }

  function hasAnyNodeMarker(text) {
    const value = (text || "").toString();
    return NODE_MARKER_VARIANTS.some(
      ([markerLeft, markerRight]) =>
        value.includes(markerLeft) || value.includes(markerRight)
    );
  }

  function extractMarkedSpanTarget(nodeText) {
    const node = (nodeText || "").toString();
    const spans = extractMarkedSpans(node);
    if (spans.length) {
      return {
        focus: "marked_subspan",
        text: spans.join(" "),
      };
    }
    return {
      focus: "node",
      text: node.trim(),
    };
  }

  function markNodeInContextForEstimate(left, node, right) {
    const leftRaw = (left || "").toString();
    const nodeRaw = (node || "").toString();
    const rightRaw = (right || "").toString();
    const leftPart = leftRaw.replace(/\s+$/u, "");
    const rightPart = rightRaw.replace(/^\s+/u, "");
    const leftSep =
      leftPart.length === 0 ? "" : leftRaw.endsWith(" ") || leftRaw.endsWith("\n") ? "" : " ";
    const rightSep =
      rightPart.length === 0 ? "" : rightRaw.startsWith(" ") || rightRaw.startsWith("\n") ? "" : " ";
    const nodePart = hasAnyNodeMarker(nodeRaw)
      ? nodeRaw
      : `${NODE_MARKER_LEFT}${nodeRaw}${NODE_MARKER_RIGHT}`;
    return `${leftPart}${leftSep}${nodePart}${rightSep}${rightPart}`.trim();
  }

  function buildEstimateTargetPayload(layout, row) {
    const leftContext = (row.leftContext || "").toString();
    const node = (row.node || "").toString();
    const rightContext = (row.rightContext || "").toString();
    const info = (row.info || "").toString();
    const target = extractMarkedSpanTarget(node);
    const classificationNote =
      target.focus === "marked_subspan"
        ? "Classify only the marked sub-span; use the rest of the node plus contexts as supporting evidence."
        : "Classify the entire node; left/right contexts simply provide supporting evidence.";

    if (layout === "compact") {
      const compactPayload = {
        left_context: leftContext,
        node,
        right_context: rightContext,
        classification_target: target,
      };
      if (info) {
        compactPayload.info = info;
      }
      return compactPayload;
    }

    return {
      left_context: leftContext,
      node,
      right_context: rightContext,
      info,
      marked_example: markNodeInContextForEstimate(leftContext, node, rightContext),
      classification_target: {
        focus: target.focus,
        text: target.text,
        note: classificationNote,
      },
    };
  }

  function estimateCachePaddingTokenContribution(paddingUnits) {
    const safeUnits = Math.max(0, Math.min(MAX_CACHE_PADDING_TOKENS, Number(paddingUnits) || 0));
    if (safeUnits <= 0) {
      return 0;
    }
    const charCount =
      CACHE_PADDING_PREFIX.length + CACHE_PADDING_TOKEN.length * safeUnits + CACHE_PADDING_SUFFIX.length;
    return estimateTokensFromChars(charCount).estimateMid;
  }

  function estimateRequiredCachePaddingUnits(sharedPrefixTokensEstimate, targetSharedPrefixTokens) {
    const baseline = Math.max(0, Number(sharedPrefixTokensEstimate) || 0);
    const target = Math.max(0, Number(targetSharedPrefixTokens) || 0);
    if (target <= 0 || baseline >= target) {
      return 0;
    }

    let low = 0;
    let high = 1;
    while (high < MAX_CACHE_PADDING_TOKENS && baseline + estimateCachePaddingTokenContribution(high) < target) {
      high *= 2;
    }
    high = Math.min(high, MAX_CACHE_PADDING_TOKENS);
    if (baseline + estimateCachePaddingTokenContribution(high) < target) {
      return MAX_CACHE_PADDING_TOKENS;
    }

    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (baseline + estimateCachePaddingTokenContribution(mid) >= target) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return low;
  }

  function buildSharedPrefixEstimate(data) {
    const config = getPromptEstimateConfig(data);
    let sharedUserPrefix = config.instructionsText;
    if (config.fewShotCount > 0) {
      sharedUserPrefix +=
        `\n\nHere are ${config.fewShotCount} labeled example(s) you should mimic when classifying:\n` +
        "[example content omitted in estimate]";
    }
    sharedUserPrefix += "\n\nNow classify this example:\n";

    const baseSharedPrefixChars = config.systemMessage.length + sharedUserPrefix.length;
    const baseTokenEstimates = estimateTokensFromChars(baseSharedPrefixChars);
    const targetRaw = (data.get("cache_pad_target_tokens") || "0").toString().trim();
    const targetSharedPrefixTokens = Math.max(0, Number.parseInt(targetRaw || "0", 10) || 0);
    const paddingUnitsEstimate = estimateRequiredCachePaddingUnits(
      baseTokenEstimates.estimateMid,
      targetSharedPrefixTokens
    );
    const paddedSharedPrefixTokensEstimate =
      baseTokenEstimates.estimateMid + estimateCachePaddingTokenContribution(paddingUnitsEstimate);
    return {
      layout: config.layout,
      fewShotCount: config.fewShotCount,
      baseSharedPrefixChars,
      baseEstimateMid: baseTokenEstimates.estimateMid,
      baseEstimateLow: baseTokenEstimates.estimateLow,
      baseEstimateHigh: baseTokenEstimates.estimateHigh,
      targetSharedPrefixTokens,
      paddingUnitsEstimate,
      paddedSharedPrefixTokensEstimate,
    };
  }

  function getPromptPreviewRow(ctx) {
    if (Array.isArray(ctx.sampledEstimateRows) && ctx.sampledEstimateRows.length > 0) {
      return {
        row: ctx.sampledEstimateRows[0],
        sampled: true,
      };
    }
    return {
      row: PROMPT_PREVIEW_PLACEHOLDER_ROW,
      sampled: false,
    };
  }

  function buildPromptPreview(data, ctx) {
    const config = getPromptEstimateConfig(data);
    const previewRowInfo = getPromptPreviewRow(ctx);
    let userContent = config.instructionsText;
    if (config.fewShotCount > 0) {
      userContent +=
        `\n\nHere are ${config.fewShotCount} labeled example(s) you should mimic when classifying:\n` +
        "[few-shot examples are selected at runtime and are omitted from GUI preview]";
    }
    userContent += "\n\nNow classify this example:\n";
    userContent += serializePayloadForEstimate(
      config.layout,
      buildEstimateTargetPayload(config.layout, previewRowInfo.row)
    );
    const suppressSystemMessage = Boolean(
      (data.get("gemini_cached_content") || "").toString().trim()
    );
    return {
      systemMessage: suppressSystemMessage
        ? "[omitted: system message is not sent when --gemini_cached_content is set]"
        : config.systemMessage,
      userMessage: userContent,
      sampled: previewRowInfo.sampled,
      fewShotCount: config.fewShotCount,
    };
  }

  function updatePromptPreview(ctx, data) {
    if (!ctx.promptPreviewSystem || !ctx.promptPreviewUser || !ctx.promptPreviewMeta) {
      return;
    }
    const preview = buildPromptPreview(data, ctx);
    ctx.promptPreviewSystem.textContent = preview.systemMessage;
    ctx.promptPreviewUser.textContent = preview.userMessage;

    let metaText = preview.sampled
      ? "Preview row source: first sampled CSV data row."
      : "Preview row source: placeholder values (sample CSV to insert first data row).";
    if (ctx.sampledEstimateSource) {
      metaText += ` Source: ${ctx.sampledEstimateSource}.`;
    }
    if (preview.fewShotCount > 0) {
      metaText += ` Few-shot examples configured (${preview.fewShotCount}) but resolved at runtime.`;
    }
    ctx.promptPreviewMeta.textContent = metaText;
  }

  function setPromptEstimateStatus(ctx, message, isError = false) {
    if (!ctx.promptTokenEstimateStatus) {
      return;
    }
    ctx.promptTokenEstimateStatus.textContent = message || "";
    ctx.promptTokenEstimateStatus.style.color = isError ? "#b00020" : "#35526d";
  }

  function updatePromptTokenEstimate(ctx, data) {
    if (!ctx.promptTokenEstimate) {
      return;
    }
    const estimate = buildSharedPrefixEstimate(data);
    const fewShotWarning =
      estimate.fewShotCount > 0
        ? ` Few-shot example text is omitted (configured count: ${estimate.fewShotCount}).`
        : "";
    const sampled =
      Array.isArray(ctx.sampledEstimateRows) && ctx.sampledEstimateRows.length > 0
        ? `<br>Sampled rows: ${ctx.sampledEstimateRows.length} (does not change shared-prefix estimate).`
        : "";
    const targetNote =
      estimate.targetSharedPrefixTokens > 0
        ? `<br>Configured target shared prefix: <code>${estimate.targetSharedPrefixTokens}</code> tokens. ` +
          `Estimated padding units: <code>${estimate.paddingUnitsEstimate}</code>. ` +
          `Estimated padded shared prefix: ~${estimate.paddedSharedPrefixTokensEstimate} tokens.`
        : "";
    const cacheHintTokens =
      estimate.targetSharedPrefixTokens > 0
        ? estimate.targetSharedPrefixTokens
        : PROMPT_CACHE_HINT_TOKENS;
    const cacheHintStatus =
      estimate.paddedSharedPrefixTokensEstimate >= cacheHintTokens
        ? "target reached by estimate"
        : "target not reached by estimate";

    let estimateHtml =
      `<strong>Shared prefix estimate (cacheable part only):</strong> ` +
      `~${estimate.baseEstimateMid} tokens ` +
      `(range ${estimate.baseEstimateLow}-${estimate.baseEstimateHigh}), ` +
      `${estimate.baseSharedPrefixChars} chars.` +
      `<br>Row-specific payload fields are excluded from this estimate.` +
      `<br>Cache hint threshold: ~${cacheHintTokens}+ tokens (${cacheHintStatus}).` +
      `<br>Layout: <code>${escapeHtml(estimate.layout)}</code>.` +
      targetNote +
      sampled;
    if (ctx.sampledEstimateSource) {
      estimateHtml += `<br>Sample source: <code>${escapeHtml(ctx.sampledEstimateSource)}</code>.`;
    }
    estimateHtml += fewShotWarning;
    ctx.promptTokenEstimate.innerHTML = estimateHtml;
  }

  function parseDelimitedRows(text, delimiter) {
    const rows = [];
    let currentRow = [];
    let currentCell = "";
    let inQuotes = false;
    let index = 0;
    while (index < text.length) {
      const char = text[index];
      if (char === '"') {
        if (inQuotes && text[index + 1] === '"') {
          currentCell += '"';
          index += 2;
          continue;
        }
        inQuotes = !inQuotes;
        index += 1;
        continue;
      }
      if (!inQuotes && char === delimiter) {
        currentRow.push(currentCell);
        currentCell = "";
        index += 1;
        continue;
      }
      if (!inQuotes && (char === "\n" || char === "\r")) {
        currentRow.push(currentCell);
        rows.push(currentRow);
        currentRow = [];
        currentCell = "";
        if (char === "\r" && text[index + 1] === "\n") {
          index += 2;
        } else {
          index += 1;
        }
        continue;
      }
      currentCell += char;
      index += 1;
    }
    if (currentCell.length > 0 || currentRow.length > 0) {
      currentRow.push(currentCell);
      rows.push(currentRow);
    }
    return rows;
  }

  function normalizeHeaderName(value) {
    return (value || "")
      .toString()
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
  }

  function parseCsvExamplesForEstimate(text) {
    const content = (text || "").replace(/^\uFEFF/, "");
    if (!content.trim()) {
      return [];
    }
    let rows = parseDelimitedRows(content, ";");
    if (rows.length > 0 && rows[0].length <= 1) {
      rows = parseDelimitedRows(content, ",");
    }
    if (!rows.length) {
      return [];
    }
    const headers = rows[0].map((value) => value.trim());
    const headerIndex = {};
    headers.forEach((header, idx) => {
      const normalized = normalizeHeaderName(header);
      if (normalized && !Object.prototype.hasOwnProperty.call(headerIndex, normalized)) {
        headerIndex[normalized] = idx;
      }
    });
    const leftIdx = headerIndex.leftcontext;
    const nodeIdx = headerIndex.node;
    const rightIdx = headerIndex.rightcontext;
    const infoIdx =
      headerIndex.info !== undefined
        ? headerIndex.info
        : headerIndex.metadata !== undefined
          ? headerIndex.metadata
          : null;
    if (nodeIdx === undefined) {
      throw new Error("CSV must include a node column.");
    }

    const records = [];
    for (let idx = 1; idx < rows.length; idx += 1) {
      const row = rows[idx];
      if (!Array.isArray(row) || row.every((cell) => !String(cell || "").trim())) {
        continue;
      }
      records.push({
        leftContext: row[leftIdx] || "",
        node: row[nodeIdx] || "",
        rightContext: row[rightIdx] || "",
        info: infoIdx === null ? "" : row[infoIdx] || "",
      });
    }
    return records;
  }

  function sampleEvenly(records, sampleSize) {
    if (!Array.isArray(records) || records.length === 0) {
      return [];
    }
    if (records.length <= sampleSize) {
      return records.slice();
    }
    const sampled = [];
    const step = records.length / sampleSize;
    for (let idx = 0; idx < sampleSize; idx += 1) {
      sampled.push(records[Math.floor(idx * step)]);
    }
    return sampled;
  }

  async function readCsvTextForEstimate(ctx) {
    if (ctx.estimateCsvFileInput && ctx.estimateCsvFileInput.files && ctx.estimateCsvFileInput.files.length > 0) {
      const csvFile = ctx.estimateCsvFileInput.files[0];
      const text = await csvFile.text();
      return { text, sourceLabel: csvFile.name || "selected file" };
    }
    const inputRaw = (ctx.form.elements.namedItem("input_path")?.value || "").toString();
    const inputPaths = inputRaw
      .split(/\r?\n/)
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0);
    if (!inputPaths.length) {
      throw new Error("No input CSV path is set. Provide input path or choose a local file.");
    }
    const firstPath = inputPaths[0];
    if (/^[a-zA-Z]:[\\/]/.test(firstPath)) {
      throw new Error("Local absolute paths cannot be read by browser fetch. Use the file picker.");
    }
    const response = await fetch(firstPath, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load "${firstPath}" (HTTP ${response.status}). Use the file picker.`);
    }
    const text = await response.text();
    return { text, sourceLabel: firstPath };
  }

  async function sampleCsvForPromptEstimate(ctx) {
    setPromptEstimateStatus(ctx, "Sampling CSV rows for token estimate...");
    if (ctx.sampleCsvButton) {
      ctx.sampleCsvButton.disabled = true;
    }
    try {
      const { text, sourceLabel } = await readCsvTextForEstimate(ctx);
      const records = parseCsvExamplesForEstimate(text);
      if (!records.length) {
        throw new Error("CSV has no data rows after header.");
      }
      ctx.sampledEstimateRows = sampleEvenly(records, PROMPT_ESTIMATE_SAMPLE_LIMIT);
      ctx.sampledEstimateSource = sourceLabel;
      const data = new FormData(ctx.form);
      updatePromptTokenEstimate(ctx, data);
      updatePromptPreview(ctx, data);
      setPromptEstimateStatus(
        ctx,
        `Sampled ${ctx.sampledEstimateRows.length}/${records.length} rows from ${sourceLabel}.`
      );
    } catch (error) {
      ctx.sampledEstimateRows = null;
      ctx.sampledEstimateSource = "";
      const data = new FormData(ctx.form);
      updatePromptTokenEstimate(ctx, data);
      updatePromptPreview(ctx, data);
      setPromptEstimateStatus(
        ctx,
        error && error.message ? error.message : "CSV sampling failed.",
        true
      );
    } finally {
      if (ctx.sampleCsvButton) {
        ctx.sampleCsvButton.disabled = false;
      }
    }
  }

  function encodeSystemPromptForCli(value) {
    if (typeof value !== "string" || !value.length) {
      return "";
    }
    const normalized = value.replace(/\r\n?/g, "\n");
    if (window.TextEncoder) {
      const encoder = new TextEncoder();
      const bytes = encoder.encode(normalized);
      let binary = "";
      bytes.forEach((byte) => {
        binary += String.fromCharCode(byte);
      });
      return window.btoa(binary);
    }
    return window.btoa(unescape(encodeURIComponent(normalized)));
  }

  function promptNeedsEncoding(value) {
    if (typeof value !== "string" || !value.length) {
      return false;
    }
    return /[\r\n]/.test(value);
  }

  function ensureProviderRegistered(ctx, slug, labelHint) {
    if (!ctx.providerSelect || !slug) {
      return;
    }
    const upper = (labelHint || slug.toUpperCase().replace(/[^A-Z0-9]+/g, "_")).toUpperCase();
    if (!providerDefaults[slug]) {
      providerDefaults[slug] = {
        apiKeyVar: `${upper}_API_KEY`,
        apiBaseVar: `${upper}_BASE_URL`,
      };
    }
    if (!Array.from(ctx.providerSelect.options).some((opt) => opt.value === slug)) {
      const option = document.createElement("option");
      option.value = slug;
      option.textContent = knownProviderLabels[slug] || `${upper.replace(/_/g, " ")} (catalog)`;
      ctx.providerSelect.appendChild(option);
    }
  }

  function setRefreshStatus(ctx, message, isError = false) {
    if (!ctx.refreshModelsStatus) {
      return;
    }
    ctx.refreshModelsStatus.textContent = message || "";
    ctx.refreshModelsStatus.style.color = isError ? "#b00020" : "";
  }

  function updateModelOptionsForProvider(ctx, provider) {
    if (!ctx.modelList) {
      return;
    }
    ctx.modelList.innerHTML = "";
    const entry = ctx.modelCatalog[provider];
    const models = getProviderCatalogModels(ctx, provider);
    if (models.length) {
      models.forEach((modelId) => {
        const option = document.createElement("option");
        option.value = modelId;
        const pricingSummary = summarizeModelPricing(ctx, provider, modelId);
        if (pricingSummary) {
          option.label = `${modelId} | ${pricingSummary}`;
          option.textContent = option.label;
        }
        ctx.modelList.appendChild(option);
      });
    }
    updateModelCatalogMeta(ctx, provider, models);
  }

  function syncProvidersFromCatalog(ctx) {
    if (!ctx.providerSelect || !ctx.modelCatalog || typeof ctx.modelCatalog !== "object") {
      return;
    }
    const catalogEntries = Object.entries(ctx.modelCatalog).filter(
      ([providerSlug]) => typeof providerSlug === "string" && providerSlug.trim().length > 0
    );
    if (!catalogEntries.length) {
      return;
    }

    const catalogProviderSet = new Set(catalogEntries.map(([providerSlug]) => providerSlug));
    const selectedBefore = ctx.providerSelect.value;

    for (const option of Array.from(ctx.providerSelect.options)) {
      if (!catalogProviderSet.has(option.value)) {
        option.remove();
      }
    }

    for (const [providerSlug, entry] of catalogEntries) {
      ensureProviderRegistered(ctx, providerSlug, providerSlug);
      const prefix = providerSlugToEnvPrefix(providerSlug);
      const apiKeyVar =
        entry && typeof entry.api_key_var === "string" && entry.api_key_var.trim()
          ? entry.api_key_var.trim()
          : `${prefix}_API_KEY`;
      const apiBaseVar =
        entry && typeof entry.api_base_var === "string" && entry.api_base_var.trim()
          ? entry.api_base_var.trim()
          : `${prefix}_BASE_URL`;
      providerDefaults[providerSlug] = { apiKeyVar, apiBaseVar };

      const option = Array.from(ctx.providerSelect.options).find((item) => item.value === providerSlug);
      if (option) {
        option.textContent = knownProviderLabels[providerSlug] || `${prefix.replace(/_/g, " ")} (catalog)`;
      }
    }

    if (
      selectedBefore &&
      Array.from(ctx.providerSelect.options).some((option) => option.value === selectedBefore)
    ) {
      ctx.providerSelect.value = selectedBefore;
    } else if (ctx.providerSelect.options.length > 0) {
      ctx.providerSelect.value = ctx.providerSelect.options[0].value;
    }
  }

  function loadModelCatalogScript() {
    const cacheBustedSrc = `config_models.js?ts=${Date.now()}`;
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = cacheBustedSrc;
      script.async = true;
      script.onload = () => {
        script.remove();
        resolve();
      };
      script.onerror = () => {
        script.remove();
        reject(new Error(`Unable to load ${cacheBustedSrc}`));
      };
      document.head.appendChild(script);
    });
  }

  function loadPricingCatalogScript() {
    const cacheBustedSrc = `config_prices.js?ts=${Date.now()}`;
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = cacheBustedSrc;
      script.async = true;
      script.onload = () => {
        script.remove();
        resolve();
      };
      script.onerror = () => {
        script.remove();
        reject(new Error(`Unable to load ${cacheBustedSrc}`));
      };
      document.head.appendChild(script);
    });
  }

  async function refreshModelCatalogFromScript(ctx) {
    if (ctx.refreshModelsButton) {
      ctx.refreshModelsButton.disabled = true;
      ctx.refreshModelsButton.textContent = "Refreshing...";
    }
    setRefreshStatus(ctx, "Refreshing provider, model, and pricing data from config_models.js/config_prices.js...");
    try {
      await loadModelCatalogScript();
      ctx.modelCatalog = getWindowModelCatalog();
      try {
        await loadPricingCatalogScript();
      } catch (pricingError) {
        console.warn(pricingError);
      }
      ctx.priceCatalog = getWindowPricingCatalog();
      syncProvidersFromCatalog(ctx);
      updatePlaceholdersForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
      updateModelOptionsForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
      updateContextualVisibility(ctx);
      handleFormChange(ctx);
      setRefreshStatus(ctx, "Provider, model, and pricing data refreshed.");
    } catch (error) {
      console.warn(error);
      ctx.modelCatalog = getWindowModelCatalog();
      ctx.priceCatalog = getWindowPricingCatalog();
      syncProvidersFromCatalog(ctx);
      updatePlaceholdersForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
      updateModelOptionsForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
      updateContextualVisibility(ctx);
      setRefreshStatus(
        ctx,
        "Could not reload config_models.js/config_prices.js; showing the currently loaded catalog.",
        true
      );
    } finally {
      if (ctx.refreshModelsButton) {
        ctx.refreshModelsButton.disabled = false;
        ctx.refreshModelsButton.textContent = "Refresh Model List";
      }
    }
  }

  function updatePlaceholdersForProvider(ctx, provider) {
    const defaults = providerDefaults[provider] || providerDefaults.openai;
    if (ctx.apiKeyVarInput && !ctx.apiKeyVarInput.value) {
      ctx.apiKeyVarInput.placeholder = defaults.apiKeyVar;
    }
    if (ctx.apiBaseVarInput && !ctx.apiBaseVarInput.value) {
      ctx.apiBaseVarInput.placeholder = defaults.apiBaseVar;
    }
  }

  function getMappedHelpText(element) {
    if (!element || !(element instanceof HTMLElement)) {
      return "";
    }
    const mapped =
      (element.id && inputHelpTextById[element.id]) ||
      (element.name && inputHelpTextById[element.name]) ||
      "";
    return typeof mapped === "string" ? mapped.trim() : "";
  }

  function closeAllFieldHelp(exceptLabel = null) {
    document.querySelectorAll('label[data-field-help="true"]').forEach((label) => {
      if (!(label instanceof HTMLLabelElement) || label === exceptLabel) {
        return;
      }
      const button = label.querySelector(".field-help-button");
      const popover = label.querySelector(".field-help-popover");
      if (!(button instanceof HTMLButtonElement) || !(popover instanceof HTMLElement)) {
        return;
      }
      button.setAttribute("aria-expanded", "false");
      popover.hidden = true;
    });
  }

  function bindFieldHelpDismissListeners(ctx) {
    if (ctx.fieldHelpDismissListenersBound) {
      return;
    }
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (target instanceof Element && target.closest(".field-help-button, .field-help-popover")) {
        return;
      }
      closeAllFieldHelp();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeAllFieldHelp();
      }
    });
    ctx.fieldHelpDismissListenersBound = true;
  }

  function resolveFieldAnchorNode(label, element) {
    if (!(label instanceof HTMLLabelElement) || !(element instanceof HTMLElement)) {
      return null;
    }
    let current = element;
    while (current.parentElement && current.parentElement !== label) {
      current = current.parentElement;
    }
    return current.parentElement === label ? current : null;
  }

  function resolveFieldHelpHeading(label, element) {
    if (!(label instanceof HTMLLabelElement) || !(element instanceof HTMLElement)) {
      return null;
    }
    const directHeading = Array.from(label.children).find((child) => {
      return (
        child instanceof HTMLElement &&
        child.tagName === "SPAN" &&
        (child.classList.contains("label-heading") || child.classList.contains("field-help-heading"))
      );
    });
    if (directHeading instanceof HTMLElement) {
      directHeading.classList.add("field-help-heading");
      return directHeading;
    }

    const directSpan = Array.from(label.children).find((child) => {
      return child instanceof HTMLElement && child.tagName === "SPAN";
    });
    if (directSpan instanceof HTMLElement) {
      directSpan.classList.add("field-help-heading");
      return directSpan;
    }

    const anchorNode = resolveFieldAnchorNode(label, element);
    if (!(anchorNode instanceof Node)) {
      return null;
    }
    const heading = document.createElement("span");
    heading.className = "field-help-heading";
    const leadingNodes = [];
    for (const node of Array.from(label.childNodes)) {
      if (node === anchorNode) {
        break;
      }
      leadingNodes.push(node);
    }
    if (!leadingNodes.some((node) => node.nodeType !== Node.TEXT_NODE || node.textContent.trim())) {
      return null;
    }
    label.insertBefore(heading, anchorNode);
    leadingNodes.forEach((node) => {
      if (node.nodeType === Node.TEXT_NODE && !node.textContent.trim()) {
        node.remove();
        return;
      }
      heading.appendChild(node);
    });
    return heading;
  }

  function enhanceFieldHelp(label, element, helpText) {
    if (!(label instanceof HTMLLabelElement) || !(element instanceof HTMLElement) || !helpText) {
      return;
    }
    if (label.dataset.fieldHelp === "true") {
      return;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "field-help-button";
    button.textContent = "?";
    button.title = helpText;
    button.setAttribute("aria-expanded", "false");

    const headingText = label.textContent ? label.textContent.replace(/\s+/g, " ").trim() : "";
    button.setAttribute(
      "aria-label",
      headingText ? `Show help for ${headingText}` : "Show field help"
    );

    const popover = document.createElement("div");
    popover.className = "field-help-popover";
    popover.hidden = true;
    popover.textContent = helpText;
    popover.id = `field-help-${fieldHelpIdCounter += 1}`;
    button.setAttribute("aria-controls", popover.id);

    if (label.classList.contains("inline")) {
      label.appendChild(button);
      label.appendChild(popover);
    } else {
      const heading = resolveFieldHelpHeading(label, element);
      if (heading instanceof HTMLElement) {
        heading.appendChild(button);
        const anchorNode = resolveFieldAnchorNode(label, element);
        if (anchorNode instanceof Node) {
          label.insertBefore(popover, anchorNode);
        } else {
          label.appendChild(popover);
        }
      } else {
        label.insertBefore(button, label.firstChild);
        label.insertBefore(popover, element);
      }
    }

    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const willOpen = button.getAttribute("aria-expanded") !== "true";
      closeAllFieldHelp(label);
      button.setAttribute("aria-expanded", willOpen ? "true" : "false");
      popover.hidden = !willOpen;
    });

    label.dataset.fieldHelp = "true";
  }

  function applyInputHoverHelp(ctx) {
    if (!ctx.form) {
      return;
    }
    bindFieldHelpDismissListeners(ctx);
    Array.from(ctx.form.elements).forEach((element) => {
      if (!element || !(element instanceof HTMLElement)) {
        return;
      }
      if (element.type === "file") {
        return;
      }
      const mapped = getMappedHelpText(element);
      if (mapped) {
        element.title = mapped;
        if (
          (element instanceof HTMLInputElement ||
            element instanceof HTMLSelectElement ||
            element instanceof HTMLTextAreaElement) &&
          element.type !== "hidden"
        ) {
          const label = element.closest("label");
          if (label instanceof HTMLLabelElement) {
            enhanceFieldHelp(label, element, mapped);
          }
        }
      }
    });
    if (ctx.refreshModelsButton) {
      ctx.refreshModelsButton.title =
        "Reload providers, model lists, and pricing directly from config_models.js/config_prices.js without reloading the page.";
    }
    if (ctx.copyButton) {
      ctx.copyButton.title = "Copy the currently generated command line to clipboard.";
    }
    if (ctx.resetButton) {
      ctx.resetButton.title = "Reset all GUI fields to defaults and clear saved settings.";
    }
    if (ctx.modelClearButton) {
      ctx.modelClearButton.title = "Clear the model input field.";
    }
    if (ctx.sampleCsvButton) {
      ctx.sampleCsvButton.title =
        "Manually re-sample CSV and refresh prompt token estimate (also runs automatically on file selection).";
    }
    if (ctx.estimateCsvFileInput) {
      ctx.estimateCsvFileInput.title =
        "Optional local CSV file used only for prompt-token estimation.";
    }
    if (ctx.sidebarToggleButton) {
      ctx.sidebarToggleButton.title =
        "Toggle the generated command sidebar between collapsed and expanded states.";
    }
  }

  function saveConfig(ctx) {
    if (!ctx.form) {
      return;
    }
    const config = {};
    Array.from(ctx.form.elements).forEach((element) => {
      if (!element.name || element.type === "file") {
        return;
      }
      if (element.type === "checkbox") {
        config[element.name] = element.checked;
      } else {
        config[element.name] = element.value;
      }
    });
    if (isModeFirstVariant(ctx.variant)) {
      config.__active_mode = normalizeMode(ctx.activeMode);
    }
    try {
      localStorage.setItem(ctx.storageKey, JSON.stringify(config));
    } catch (error) {
      console.warn("Unable to persist configuration:", error);
    }
  }

  function loadConfig(ctx) {
    try {
      const stored = localStorage.getItem(ctx.storageKey);
      if (!stored) {
        return;
      }
      const config = JSON.parse(stored);
      if (isModeFirstVariant(ctx.variant) && config.__active_mode) {
        ctx.activeMode = normalizeMode(config.__active_mode);
      }
      for (const [name, value] of Object.entries(config)) {
        if (name === "__active_mode") {
          continue;
        }
        const control = ctx.form.elements.namedItem(name);
        if (!control || control.type === "file") continue;
        if (name === "provider" && typeof value === "string") {
          const hasCatalogProviders =
            ctx.modelCatalog && typeof ctx.modelCatalog === "object" && Object.keys(ctx.modelCatalog).length > 0;
          if (hasCatalogProviders && !Object.prototype.hasOwnProperty.call(ctx.modelCatalog, value)) {
            continue;
          }
          ensureProviderRegistered(ctx, value, value);
        }
        if (control instanceof RadioNodeList) {
          continue;
        }
        if (control.type === "checkbox") {
          control.checked = Boolean(value);
        } else {
          control.value = value;
        }
      }
    } catch (error) {
      console.warn("Unable to load stored configuration:", error);
    }
  }

  function isGeminiTarget(provider, modelValue) {
    const model = (modelValue || "").toString().trim().toLowerCase();
    return provider === "google" || model.includes("gemini");
  }

  function isClaudeTarget(provider, modelValue) {
    const model = (modelValue || "").toString().trim().toLowerCase();
    return provider === "anthropic" || model.includes("claude");
  }

  function matchesProviderVisibility(rule, provider, modelValue) {
    const tokens = (rule || "")
      .toString()
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    if (!tokens.length) {
      return true;
    }
    return tokens.some((token) => {
      if (token === "vertex") return provider === "vertex";
      if (token === "requesty") return provider === "requesty";
      if (token === "gemini") return isGeminiTarget(provider, modelValue);
      if (token === "claude") return isClaudeTarget(provider, modelValue);
      return provider === token;
    });
  }

  function applyProviderVisibility(ctx) {
    const provider = ctx.providerSelect?.value || defaultValues.provider;
    const modelValue = ctx.modelInput?.value || "";
    document.querySelectorAll("[data-provider-visible]").forEach((element) => {
      const shouldShow = matchesProviderVisibility(
        element.getAttribute("data-provider-visible"),
        provider,
        modelValue
      );
      element.hidden = !shouldShow;
    });
    document.querySelectorAll("[data-provider-hidden]").forEach((element) => {
      const shouldHide = matchesProviderVisibility(
        element.getAttribute("data-provider-hidden"),
        provider,
        modelValue
      );
      element.hidden = shouldHide;
    });
  }

  function applyModeVisibility(ctx) {
    const mode = normalizeMode(ctx.activeMode);
    document.body.dataset.commandMode = mode;
    document.querySelectorAll("[data-mode-visible]").forEach((element) => {
      const allowedModes = element
        .getAttribute("data-mode-visible")
        .split(/\s+/)
        .filter(Boolean);
      element.hidden = !allowedModes.includes(mode);
    });
    document.querySelectorAll("[data-mode-hidden]").forEach((element) => {
      const hiddenModes = element
        .getAttribute("data-mode-hidden")
        .split(/\s+/)
        .filter(Boolean);
      element.hidden = hiddenModes.includes(mode);
    });
    if (ctx.modeButtons && ctx.modeButtons.length > 0) {
      ctx.modeButtons.forEach((button) => {
        const buttonMode = normalizeMode(button.dataset.modeChoice);
        const isActive = buttonMode === mode;
        button.classList.toggle("is-active", isActive);
        button.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
    }
    if (ctx.modeLabelElements && ctx.modeLabelElements.length > 0) {
      ctx.modeLabelElements.forEach((element) => {
        element.textContent = getModeLabel(mode);
      });
    }
    updateModeSpecificFieldStates(ctx, mode);
  }

  function updateModeSpecificFieldStates(ctx, mode) {
    const normalizedMode = normalizeMode(mode);
    if (ctx.outputPathModeBadge) {
      ctx.outputPathModeBadge.textContent = normalizedMode === "resume" ? "Required" : "Optional";
    }
    if (ctx.outputPathModeHint) {
      ctx.outputPathModeHint.textContent =
        normalizedMode === "resume"
          ? "Point to the existing output CSV that should be resumed in place."
          : "Leave blank for the default generated output path.";
    }
    if (ctx.outputPathInput) {
      ctx.outputPathInput.required = normalizedMode === "resume";
      ctx.outputPathInput.setAttribute(
        "aria-required",
        normalizedMode === "resume" ? "true" : "false"
      );
    }
    if (ctx.inputPathInput) {
      const inputRequired = normalizedMode !== "resume";
      ctx.inputPathInput.required = inputRequired;
      ctx.inputPathInput.setAttribute("aria-required", inputRequired ? "true" : "false");
    }
    if (ctx.logprobsInput) {
      const logprobsEnabled = normalizedMode !== "metrics";
      ctx.logprobsInput.disabled = !logprobsEnabled;
      ctx.logprobsInput.setAttribute("aria-disabled", logprobsEnabled ? "false" : "true");
    }
  }

  function updateSidebarSummary(ctx, commandMeta) {
    if (!ctx.sidebarModeLabel && !ctx.sidebarFlagCount) {
      return;
    }
    const modeLabel = getModeLabel(commandMeta.mode || ctx.activeMode);
    if (ctx.sidebarModeLabel) {
      ctx.sidebarModeLabel.textContent = modeLabel;
    }
    if (ctx.sidebarFlagCount) {
      const count = Number(commandMeta.nonDefaultFlagCount) || 0;
      ctx.sidebarFlagCount.textContent = `${count} non-default flag${count === 1 ? "" : "s"}`;
    }
  }

  function updateSidebarState(ctx, collapsed, persist = true) {
    if (!ctx.commandPanel || !ctx.sidebarToggleButton || !isModeFirstVariant(ctx.variant)) {
      return;
    }
    ctx.sidebarCollapsed = Boolean(collapsed);
    ctx.commandPanel.classList.toggle("is-collapsed", ctx.sidebarCollapsed);
    ctx.sidebarToggleButton.textContent = ctx.sidebarCollapsed ? "Expand" : "Collapse";
    ctx.sidebarToggleButton.setAttribute("aria-expanded", ctx.sidebarCollapsed ? "false" : "true");
    const uiStorageKey = getModeFirstUiStorageKey(ctx.variant);
    if (persist && uiStorageKey) {
      try {
        localStorage.setItem(uiStorageKey, JSON.stringify({ sidebarCollapsed: ctx.sidebarCollapsed }));
      } catch (error) {
        console.warn("Unable to persist mode-first UI state:", error);
      }
    }
  }

  function loadSidebarState(ctx) {
    const uiStorageKey = getModeFirstUiStorageKey(ctx.variant);
    if (!uiStorageKey) {
      return true;
    }
    try {
      const stored = localStorage.getItem(uiStorageKey);
      if (!stored) {
        return true;
      }
      const parsed = JSON.parse(stored);
      return parsed.sidebarCollapsed !== false;
    } catch (error) {
      console.warn("Unable to load mode-first UI state:", error);
      return true;
    }
  }

  function updateContextualVisibility(ctx) {
    const provider = ctx.providerSelect?.value || defaultValues.provider;
    if (ctx.vertexAuthOptions) {
      ctx.vertexAuthOptions.style.display = provider === "vertex" ? "" : "none";
    }
    if (ctx.createGeminiCacheCheckbox) {
      window.toggleGeminiCacheMode(Boolean(ctx.createGeminiCacheCheckbox.checked));
    }
    if (isModeFirstVariant(ctx.variant)) {
      applyModeVisibility(ctx);
      applyProviderVisibility(ctx);
    }
  }

  window.toggleGeminiCacheMode = function toggleGeminiCacheMode(createEnabled) {
    const createOptions = document.getElementById("gemini_cache_create_options");
    const manualOptions = document.getElementById("gemini_cache_manual");
    if (createOptions) {
      createOptions.style.display = createEnabled ? "" : "none";
    }
    if (manualOptions) {
      manualOptions.style.display = createEnabled ? "none" : "";
    }
  };

  function createCommandAccumulator() {
    const args = ["python", "benchmark_agent.py"];
    const flags = [];
    return {
      args,
      flags,
      pushFlag(flag, ...values) {
        args.push(flag, ...values);
        flags.push(flag);
      },
      pushRaw(...values) {
        args.push(...values);
      },
    };
  }

  function buildCommandClassic(data) {
    const command = createCommandAccumulator();
    const metricsOnly = Boolean(data.get("metrics_only"));

    const inputRaw = (data.get("input_path") ?? "").toString();
    const inputPaths = inputRaw
      .split(/\r?\n/)
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0);
    command.pushFlag("--input");
    if (inputPaths.length === 0) {
      command.pushRaw(shellQuote(""));
    } else {
      for (const inputPath of inputPaths) {
        command.pushRaw(shellQuote(inputPath));
      }
    }

    const modelValue = data.get("model")?.toString().trim() ?? "";
    command.pushFlag("--model", shellQuote(modelValue));

    const outputValue = data.get("output_path")?.toString().trim();
    if (outputValue) {
      command.pushFlag("--output", shellQuote(outputValue));
    }
    if (data.get("reprompt_unclassified")) {
      command.pushFlag("--unclassified");
    }
    if (data.get("repeat_unclassified")) {
      command.pushFlag("--repeat_unclassified");
    }

    const taskName = data.get("task_name")?.toString().trim() ?? "";
    if (taskName) {
      command.pushFlag("--task_name", shellQuote(taskName));
    }
    const taskDescription = data.get("task_description")?.toString().trim() ?? "";
    if (taskDescription) {
      command.pushFlag("--task_description", shellQuote(taskDescription));
    }
    const tags = data.get("tags")?.toString().trim() ?? "";
    if (tags) {
      command.pushFlag("--tags", shellQuote(tags));
    }

    const provider = data.get("provider");
    if (provider && provider !== defaultValues.provider) {
      command.pushFlag("--provider", provider);
    }
    if (provider === "vertex") {
      const vertexAutoAdcLogin = Boolean(data.get("vertex_auto_adc_login"));
      if (!vertexAutoAdcLogin) {
        command.pushFlag("--no-vertex_auto_adc_login");
      }
      const vertexRefreshSeconds =
        data.get("vertex_access_token_refresh_seconds")?.toString().trim() ?? "";
      if (
        vertexRefreshSeconds &&
        vertexRefreshSeconds !== defaultValues.vertex_access_token_refresh_seconds
      ) {
        command.pushFlag("--vertex_access_token_refresh_seconds", vertexRefreshSeconds);
      }
    }

    const temperature = data.get("temperature")?.trim();
    if (temperature && temperature !== defaultValues.temperature) {
      command.pushFlag("--temperature", temperature);
    }
    const topP = data.get("top_p")?.trim();
    if (topP && topP !== defaultValues.top_p) {
      command.pushFlag("--top_p", topP);
    }
    const topK = data.get("top_k")?.trim();
    if (topK) {
      command.pushFlag("--top_k", topK);
    }
    const requestIntervalMs = data.get("request_interval_ms")?.trim();
    if (requestIntervalMs && requestIntervalMs !== defaultValues.request_interval_ms) {
      command.pushFlag("--request_interval_ms", requestIntervalMs);
    }
    const threads = data.get("threads")?.toString().trim() ?? "";
    if (threads && threads !== defaultValues.threads) {
      command.pushFlag("--threads", threads);
    }
    const promptLogDetail = data.get("prompt_log_detail");
    if (promptLogDetail && promptLogDetail !== defaultValues.prompt_log_detail) {
      command.pushFlag("--prompt_log_detail", promptLogDetail);
    }
    const flushRows = data.get("flush_rows")?.toString().trim() ?? "";
    if (flushRows && flushRows !== defaultValues.flush_rows) {
      command.pushFlag("--flush_rows", flushRows);
    }
    const flushSeconds = data.get("flush_seconds")?.toString().trim() ?? "";
    if (flushSeconds && flushSeconds !== defaultValues.flush_seconds) {
      command.pushFlag("--flush_seconds", flushSeconds);
    }
    const requestTimeoutSeconds = data.get("request_timeout_seconds")?.toString().trim() ?? "";
    if (requestTimeoutSeconds && requestTimeoutSeconds !== defaultValues.request_timeout_seconds) {
      command.pushFlag("--request_timeout_seconds", requestTimeoutSeconds);
    }
    const maxRetries = data.get("max_retries")?.toString().trim() ?? "";
    if (maxRetries && maxRetries !== defaultValues.max_retries) {
      command.pushFlag("--max_retries", maxRetries);
    }
    const retryDelay = data.get("retry_delay")?.toString().trim() ?? "";
    if (retryDelay && retryDelay !== defaultValues.retry_delay) {
      command.pushFlag("--retry_delay", retryDelay);
    }

    const serviceTier = data.get("service_tier");
    if (serviceTier && serviceTier !== defaultValues.service_tier) {
      command.pushFlag("--service_tier", serviceTier);
    }
    const reasoningEffort = data.get("reasoning_effort");
    if (reasoningEffort && reasoningEffort !== defaultValues.reasoning_effort) {
      command.pushFlag("--reasoning_effort", reasoningEffort);
    }
    const verbosity = data.get("verbosity");
    if (verbosity && verbosity !== defaultValues.verbosity) {
      command.pushFlag("--verbosity", verbosity);
    }
    const thinkingLevel = data.get("thinking_level");
    if (thinkingLevel && thinkingLevel !== defaultValues.thinking_level) {
      command.pushFlag("--thinking_level", thinkingLevel);
    }
    const effort = data.get("effort");
    if (effort && effort !== defaultValues.effort) {
      command.pushFlag("--effort", effort);
    }
    if (data.get("strict_control_acceptance")) {
      command.pushFlag("--strict_control_acceptance");
    }

    const fewShot = data.get("few_shot_examples")?.trim();
    if (fewShot && Number(fewShot) > 0) {
      command.pushFlag("--few_shot_examples", fewShot);
    }
    const promptLayout = data.get("prompt_layout");
    if (promptLayout && promptLayout !== defaultValues.prompt_layout) {
      command.pushFlag("--prompt_layout", promptLayout);
    }
    const cachePadTargetTokens = data.get("cache_pad_target_tokens")?.toString().trim() ?? "";
    if (cachePadTargetTokens && Number(cachePadTargetTokens) > 0) {
      command.pushFlag("--cache_pad_target_tokens", cachePadTargetTokens);
    }
    const promptCacheKey = data.get("prompt_cache_key")?.toString().trim() ?? "";
    if (promptCacheKey) {
      command.pushFlag("--prompt_cache_key", shellQuote(promptCacheKey));
    }
    if (data.get("requesty_auto_cache")) {
      command.pushFlag("--requesty_auto_cache");
    }

    const geminiCachedContent = data.get("gemini_cached_content")?.toString().trim() ?? "";
    if (geminiCachedContent) {
      command.pushFlag("--gemini_cached_content", shellQuote(geminiCachedContent));
    }
    const createGeminiCache = data.get("create_gemini_cache");
    if (createGeminiCache) {
      command.pushFlag("--create_gemini_cache");
      const geminiCacheTtl = data.get("gemini_cache_ttl")?.toString().trim() ?? "";
      if (geminiCacheTtl && Number(geminiCacheTtl) !== 3600) {
        command.pushFlag("--gemini_cache_ttl", geminiCacheTtl);
      }
      if (!data.get("gemini_cache_ttl_autoupdate")) {
        command.pushFlag("--no-gemini_cache_ttl_autoupdate");
      }
      if (data.get("keep_gemini_cache")) {
        command.pushFlag("--keep_gemini_cache");
      }
    }

    const labelsPath = data.get("labels_path")?.trim();
    if (labelsPath) {
      command.pushFlag("--labels", shellQuote(labelsPath));
    }
    if (metricsOnly) {
      command.pushFlag("--metrics_only");
    }

    const rawSystemPrompt = data.get("system_prompt");
    const trimmedSystemPrompt =
      typeof rawSystemPrompt === "string" ? rawSystemPrompt.trim() : "";
    if (trimmedSystemPrompt && trimmedSystemPrompt !== defaultValues.system_prompt) {
      if (promptNeedsEncoding(trimmedSystemPrompt)) {
        const serializedPrompt = encodeSystemPromptForCli(trimmedSystemPrompt);
        if (serializedPrompt) {
          command.pushFlag("--system_prompt_b64", serializedPrompt);
        }
      } else {
        command.pushFlag("--system_prompt", shellQuote(trimmedSystemPrompt));
      }
    }
    if (data.get("enable_cot")) {
      command.pushFlag("--enable_cot");
    }
    if (!data.get("include_explanations")) {
      command.pushFlag("--no_explanation");
    }
    if (!metricsOnly && data.get("logprobs")) {
      command.pushFlag("--logprobs");
    }
    if (data.get("calibration")) {
      command.pushFlag("--calibration");
    }
    if (document.getElementById("confusion_heatmap") && !data.get("confusion_heatmap")) {
      command.pushFlag("--no-confusion_heatmap");
    }

    const apiKeyVar = data.get("api_key_var")?.trim();
    if (apiKeyVar) {
      command.pushFlag("--api_key_var", shellQuote(apiKeyVar));
    }
    const apiBaseVar = data.get("api_base_var")?.trim();
    if (apiBaseVar) {
      command.pushFlag("--api_base_var", shellQuote(apiBaseVar));
    }

    if (data.get("validator_enable")) {
      const validatorCmd = data.get("validator_cmd")?.toString().trim() ?? "";
      if (validatorCmd) {
        command.pushFlag("--validator_cmd", shellQuote(validatorCmd));
        const validatorArgs = buildValidatorArgsValue(data);
        if (validatorArgs) {
          command.pushFlag("--validator_args", shellQuote(validatorArgs));
        }
        const validatorTimeout = data.get("validator_timeout")?.toString().trim() ?? "";
        if (validatorTimeout && validatorTimeout !== defaultValues.validator_timeout) {
          command.pushFlag("--validator_timeout", validatorTimeout);
        }
        const maxCandidates = data.get("validator_prompt_max_candidates")?.toString().trim() ?? "";
        if (maxCandidates && maxCandidates !== defaultValues.validator_prompt_max_candidates) {
          command.pushFlag("--validator_prompt_max_candidates", maxCandidates);
        }
        const maxChars = data.get("validator_prompt_max_chars")?.toString().trim() ?? "";
        if (maxChars && maxChars !== defaultValues.validator_prompt_max_chars) {
          command.pushFlag("--validator_prompt_max_chars", maxChars);
        }
        const exhaustedPolicy = data.get("validator_exhausted_policy");
        if (exhaustedPolicy && exhaustedPolicy !== defaultValues.validator_exhausted_policy) {
          command.pushFlag("--validator_exhausted_policy", exhaustedPolicy);
        }
        if (data.get("validator_debug")) {
          command.pushFlag("--validator_debug");
        }
      }
    }

    return {
      mode: metricsOnly ? "metrics" : "run",
      args: command.args,
      flags: command.flags.slice(),
      nonDefaultFlagCount: 0,
    };
  }

  function buildCommandPreview(ctx, data) {
    const mode = normalizeMode(ctx.activeMode);
    const command = createCommandAccumulator();
    const provider = data.get("provider");
    const modelValue = data.get("model")?.toString().trim() ?? "";
    const geminiTarget = isGeminiTarget(provider, modelValue);
    const claudeTarget = isClaudeTarget(provider, modelValue);
    const outputValue = data.get("output_path")?.toString().trim() ?? "";

    if (mode === "resume") {
      command.pushFlag("--resume");
      command.pushFlag("--output", shellQuote(outputValue));
      if (data.get("reprompt_unclassified")) {
        command.pushFlag("--unclassified");
      }
      if (data.get("repeat_unclassified")) {
        command.pushFlag("--repeat_unclassified");
      }
    } else {
      const inputRaw = (data.get("input_path") ?? "").toString();
      const inputPaths = inputRaw
        .split(/\r?\n/)
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
      command.pushFlag("--input");
      if (inputPaths.length === 0) {
        command.pushRaw(shellQuote(""));
      } else {
        inputPaths.forEach((inputPath) => command.pushRaw(shellQuote(inputPath)));
      }

      if (mode === "metrics") {
        command.pushFlag("--metrics_only");
      } else {
        command.pushFlag("--model", shellQuote(modelValue));
        if (outputValue) {
          command.pushFlag("--output", shellQuote(outputValue));
        }
        if (data.get("repeat_unclassified")) {
          command.pushFlag("--repeat_unclassified");
        }
      }

      const labelsPath = data.get("labels_path")?.trim();
      if (labelsPath) {
        command.pushFlag("--labels", shellQuote(labelsPath));
      }
      const taskName = data.get("task_name")?.toString().trim() ?? "";
      if (taskName) {
        command.pushFlag("--task_name", shellQuote(taskName));
      }
      const taskDescription = data.get("task_description")?.toString().trim() ?? "";
      if (taskDescription) {
        command.pushFlag("--task_description", shellQuote(taskDescription));
      }
      const tags = data.get("tags")?.toString().trim() ?? "";
      if (tags) {
        command.pushFlag("--tags", shellQuote(tags));
      }
      if (data.get("calibration")) {
        command.pushFlag("--calibration");
      }
      if (document.getElementById("confusion_heatmap") && !data.get("confusion_heatmap")) {
        command.pushFlag("--no-confusion_heatmap");
      }
    }

    if (mode === "run" || mode === "validator") {
      if (provider && provider !== defaultValues.provider) {
        command.pushFlag("--provider", provider);
      }
      if (provider === "vertex") {
        const vertexAutoAdcLogin = Boolean(data.get("vertex_auto_adc_login"));
        if (!vertexAutoAdcLogin) {
          command.pushFlag("--no-vertex_auto_adc_login");
        }
        const vertexRefreshSeconds =
          data.get("vertex_access_token_refresh_seconds")?.toString().trim() ?? "";
        if (
          vertexRefreshSeconds &&
          vertexRefreshSeconds !== defaultValues.vertex_access_token_refresh_seconds
        ) {
          command.pushFlag("--vertex_access_token_refresh_seconds", vertexRefreshSeconds);
        }
      }

      const rawSystemPrompt = data.get("system_prompt");
      const trimmedSystemPrompt =
        typeof rawSystemPrompt === "string" ? rawSystemPrompt.trim() : "";
      if (trimmedSystemPrompt && trimmedSystemPrompt !== defaultValues.system_prompt) {
        if (promptNeedsEncoding(trimmedSystemPrompt)) {
          const serializedPrompt = encodeSystemPromptForCli(trimmedSystemPrompt);
          if (serializedPrompt) {
            command.pushFlag("--system_prompt_b64", serializedPrompt);
          }
        } else {
          command.pushFlag("--system_prompt", shellQuote(trimmedSystemPrompt));
        }
      }
      if (data.get("enable_cot")) {
        command.pushFlag("--enable_cot");
      }
      if (!data.get("include_explanations")) {
        command.pushFlag("--no_explanation");
      }

      const fewShot = data.get("few_shot_examples")?.trim();
      if (fewShot && Number(fewShot) > 0) {
        command.pushFlag("--few_shot_examples", fewShot);
      }
      const promptLayout = data.get("prompt_layout");
      if (promptLayout && promptLayout !== defaultValues.prompt_layout) {
        command.pushFlag("--prompt_layout", promptLayout);
      }

      const temperature = data.get("temperature")?.trim();
      if (temperature && temperature !== defaultValues.temperature) {
        command.pushFlag("--temperature", temperature);
      }
      const topP = data.get("top_p")?.trim();
      if (topP && topP !== defaultValues.top_p) {
        command.pushFlag("--top_p", topP);
      }
      const topK = data.get("top_k")?.trim();
      if (topK) {
        command.pushFlag("--top_k", topK);
      }
      const requestIntervalMs = data.get("request_interval_ms")?.trim();
      if (requestIntervalMs && requestIntervalMs !== defaultValues.request_interval_ms) {
        command.pushFlag("--request_interval_ms", requestIntervalMs);
      }
      const threads = data.get("threads")?.toString().trim() ?? "";
      if (threads && threads !== defaultValues.threads) {
        command.pushFlag("--threads", threads);
      }
      const requestTimeoutSeconds = data.get("request_timeout_seconds")?.toString().trim() ?? "";
      if (requestTimeoutSeconds && requestTimeoutSeconds !== defaultValues.request_timeout_seconds) {
        command.pushFlag("--request_timeout_seconds", requestTimeoutSeconds);
      }
      const maxRetries = data.get("max_retries")?.toString().trim() ?? "";
      if (maxRetries && maxRetries !== defaultValues.max_retries) {
        command.pushFlag("--max_retries", maxRetries);
      }
      const retryDelay = data.get("retry_delay")?.toString().trim() ?? "";
      if (retryDelay && retryDelay !== defaultValues.retry_delay) {
        command.pushFlag("--retry_delay", retryDelay);
      }
      if (data.get("logprobs")) {
        command.pushFlag("--logprobs");
      }

      const promptLogDetail = data.get("prompt_log_detail");
      if (promptLogDetail && promptLogDetail !== defaultValues.prompt_log_detail) {
        command.pushFlag("--prompt_log_detail", promptLogDetail);
      }
      const flushRows = data.get("flush_rows")?.toString().trim() ?? "";
      if (flushRows && flushRows !== defaultValues.flush_rows) {
        command.pushFlag("--flush_rows", flushRows);
      }
      const flushSeconds = data.get("flush_seconds")?.toString().trim() ?? "";
      if (flushSeconds && flushSeconds !== defaultValues.flush_seconds) {
        command.pushFlag("--flush_seconds", flushSeconds);
      }
      if (data.get("strict_control_acceptance")) {
        command.pushFlag("--strict_control_acceptance");
      }

      const serviceTier = data.get("service_tier");
      if (serviceTier && serviceTier !== defaultValues.service_tier) {
        command.pushFlag("--service_tier", serviceTier);
      }
      const reasoningEffort = data.get("reasoning_effort");
      if (reasoningEffort && reasoningEffort !== defaultValues.reasoning_effort) {
        command.pushFlag("--reasoning_effort", reasoningEffort);
      }
      const verbosity = data.get("verbosity");
      if (verbosity && verbosity !== defaultValues.verbosity) {
        command.pushFlag("--verbosity", verbosity);
      }
      const thinkingLevel = data.get("thinking_level");
      if (geminiTarget && thinkingLevel && thinkingLevel !== defaultValues.thinking_level) {
        command.pushFlag("--thinking_level", thinkingLevel);
      }
      const effort = data.get("effort");
      if (claudeTarget && effort && effort !== defaultValues.effort) {
        command.pushFlag("--effort", effort);
      }

      const cachePadTargetTokens = data.get("cache_pad_target_tokens")?.toString().trim() ?? "";
      if (cachePadTargetTokens && Number(cachePadTargetTokens) > 0) {
        command.pushFlag("--cache_pad_target_tokens", cachePadTargetTokens);
      }
      const promptCacheKey = data.get("prompt_cache_key")?.toString().trim() ?? "";
      if (promptCacheKey) {
        command.pushFlag("--prompt_cache_key", shellQuote(promptCacheKey));
      }
      if (provider === "requesty" && data.get("requesty_auto_cache")) {
        command.pushFlag("--requesty_auto_cache");
      }

      const geminiCachedContent = data.get("gemini_cached_content")?.toString().trim() ?? "";
      if (geminiTarget && geminiCachedContent) {
        command.pushFlag("--gemini_cached_content", shellQuote(geminiCachedContent));
      }
      const createGeminiCache = Boolean(data.get("create_gemini_cache"));
      if (geminiTarget && createGeminiCache) {
        command.pushFlag("--create_gemini_cache");
        const geminiCacheTtl = data.get("gemini_cache_ttl")?.toString().trim() ?? "";
        if (geminiCacheTtl && Number(geminiCacheTtl) !== 3600) {
          command.pushFlag("--gemini_cache_ttl", geminiCacheTtl);
        }
        if (!data.get("gemini_cache_ttl_autoupdate")) {
          command.pushFlag("--no-gemini_cache_ttl_autoupdate");
        }
        if (data.get("keep_gemini_cache")) {
          command.pushFlag("--keep_gemini_cache");
        }
      }

      if (mode === "validator") {
        const validatorCmd = data.get("validator_cmd")?.toString().trim() ?? "";
        if (validatorCmd) {
          command.pushFlag("--validator_cmd", shellQuote(validatorCmd));
          const validatorArgs = buildValidatorArgsValue(data);
          if (validatorArgs) {
            command.pushFlag("--validator_args", shellQuote(validatorArgs));
          }
          const validatorTimeout = data.get("validator_timeout")?.toString().trim() ?? "";
          if (validatorTimeout && validatorTimeout !== defaultValues.validator_timeout) {
            command.pushFlag("--validator_timeout", validatorTimeout);
          }
          const maxCandidates = data.get("validator_prompt_max_candidates")?.toString().trim() ?? "";
          if (maxCandidates && maxCandidates !== defaultValues.validator_prompt_max_candidates) {
            command.pushFlag("--validator_prompt_max_candidates", maxCandidates);
          }
          const maxChars = data.get("validator_prompt_max_chars")?.toString().trim() ?? "";
          if (maxChars && maxChars !== defaultValues.validator_prompt_max_chars) {
            command.pushFlag("--validator_prompt_max_chars", maxChars);
          }
          const exhaustedPolicy = data.get("validator_exhausted_policy");
          if (exhaustedPolicy && exhaustedPolicy !== defaultValues.validator_exhausted_policy) {
            command.pushFlag("--validator_exhausted_policy", exhaustedPolicy);
          }
          if (data.get("validator_debug")) {
            command.pushFlag("--validator_debug");
          }
        }
      }
    }

    const baselineFlags =
      mode === "resume"
        ? new Set(["--resume", "--output"])
        : mode === "metrics"
          ? new Set(["--input", "--metrics_only"])
          : new Set(["--input", "--model"]);
    const nonDefaultFlagCount = command.flags.filter((flag) => !baselineFlags.has(flag)).length;
    return {
      mode,
      args: command.args,
      flags: command.flags.slice(),
      nonDefaultFlagCount,
    };
  }

  function renderCommand(ctx, commandResult) {
    ctx.latestCommandText = commandResult.args.join(" ");
    ctx.latestCommandMeta = commandResult;
    if (ctx.commandOutput) {
      ctx.commandOutput.innerHTML = renderCommandTokens(commandResult.args);
    }
    updateSidebarSummary(ctx, commandResult);
  }

  function handleFormChange(ctx) {
    const data = new FormData(ctx.form);
    const commandResult = isModeFirstVariant(ctx.variant)
      ? buildCommandPreview(ctx, data)
      : buildCommandClassic(data);
    renderCommand(ctx, commandResult);
    if (
      !isModeFirstVariant(ctx.variant) ||
      (normalizeMode(ctx.activeMode) !== "metrics" && normalizeMode(ctx.activeMode) !== "resume")
    ) {
      updatePromptTokenEstimate(ctx, data);
      updatePromptPreview(ctx, data);
    }
    if (!ctx.isInitializing) {
      saveConfig(ctx);
    }
  }

  function copyCommand(ctx) {
    const command = ctx.latestCommandText;
    navigator.clipboard
      .writeText(command)
      .then(() => {
        if (!ctx.copyButton) {
          return;
        }
        ctx.copyButton.textContent = "Copied!";
        setTimeout(() => {
          ctx.copyButton.textContent = ctx.copyButtonDefaultText;
        }, 1500);
      })
      .catch(() => {
        if (!ctx.copyButton) {
          return;
        }
        ctx.copyButton.textContent = "Copy failed";
        setTimeout(() => {
          ctx.copyButton.textContent = ctx.copyButtonDefaultText;
        }, 1500);
      });
  }

  function resetConfigToDefaults(ctx) {
    try {
      localStorage.removeItem(ctx.storageKey);
      const uiStorageKey = getModeFirstUiStorageKey(ctx.variant);
      if (uiStorageKey) {
        localStorage.removeItem(uiStorageKey);
      }
    } catch (error) {
      console.warn("Unable to clear stored configuration:", error);
    }

    ctx.form.reset();
    for (const [name, value] of Object.entries(defaultValues)) {
      const control = ctx.form.elements.namedItem(name);
      if (!control || control.type === "file" || control instanceof RadioNodeList) {
        continue;
      }
      if (control.type === "checkbox") {
        control.checked = Boolean(value);
      } else {
        control.value = value;
      }
    }

    if (ctx.estimateCsvFileInput) {
      ctx.estimateCsvFileInput.value = "";
    }
    ctx.sampledEstimateRows = null;
    ctx.sampledEstimateSource = "";
    setPromptEstimateStatus(ctx, "Optional: sample a CSV to include real row text in the estimate.");

    ensureProviderRegistered(ctx, defaultValues.provider, defaultValues.provider);
    if (ctx.providerSelect) {
      ctx.providerSelect.value = defaultValues.provider;
    }
    if (ctx.modelInput) {
      ctx.modelInput.value = "";
    }
    if (isModeFirstVariant(ctx.variant)) {
      ctx.activeMode = "run";
      updateSidebarState(ctx, true, false);
    }
    updatePlaceholdersForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
    updateModelOptionsForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
    updateContextualVisibility(ctx);
    handleFormChange(ctx);
  }

  function bindCommonListeners(ctx) {
    ctx.form.addEventListener("input", () => handleFormChange(ctx));
    if (ctx.copyButton) {
      ctx.copyButton.addEventListener("click", () => copyCommand(ctx));
    }
    if (ctx.resetButton) {
      ctx.resetButton.addEventListener("click", () => resetConfigToDefaults(ctx));
    }
    if (ctx.sampleCsvButton) {
      ctx.sampleCsvButton.addEventListener("click", () => sampleCsvForPromptEstimate(ctx));
    }
    if (ctx.estimateCsvFileInput) {
      ctx.estimateCsvFileInput.addEventListener("change", async () => {
        ctx.sampledEstimateRows = null;
        ctx.sampledEstimateSource = "";
        setPromptEstimateStatus(ctx, "Local CSV selected. Sampling now...");
        await sampleCsvForPromptEstimate(ctx);
      });
    }
    if (ctx.refreshModelsButton) {
      ctx.refreshModelsButton.addEventListener("click", () => refreshModelCatalogFromScript(ctx));
    }
    if (ctx.providerSelect) {
      ctx.providerSelect.addEventListener("change", () => {
        if (ctx.modelInput && normalizeMode(ctx.activeMode) !== "metrics") {
          ctx.modelInput.value = "";
        }
        updatePlaceholdersForProvider(ctx, ctx.providerSelect.value);
        updateModelOptionsForProvider(ctx, ctx.providerSelect.value);
        updateContextualVisibility(ctx);
        handleFormChange(ctx);
      });
    }
    if (ctx.serviceTierInput) {
      ctx.serviceTierInput.addEventListener("change", () => {
        updateModelOptionsForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
        updateContextualVisibility(ctx);
        handleFormChange(ctx);
      });
    }
    if (ctx.modelClearButton) {
      ctx.modelClearButton.addEventListener("click", () => {
        if (ctx.modelInput) {
          ctx.modelInput.value = "";
          ctx.modelInput.focus();
        }
        updateContextualVisibility(ctx);
        handleFormChange(ctx);
      });
    }
    if (ctx.modelInput) {
      ctx.modelInput.addEventListener("input", () => {
        updateModelCatalogMeta(ctx, ctx.providerSelect?.value || defaultValues.provider);
        updateContextualVisibility(ctx);
      });
    }
    if (ctx.createGeminiCacheCheckbox) {
      ctx.createGeminiCacheCheckbox.addEventListener("change", () => {
        window.toggleGeminiCacheMode(Boolean(ctx.createGeminiCacheCheckbox.checked));
        handleFormChange(ctx);
      });
    }
  }

  function bindPreviewModeListeners(ctx) {
    if (ctx.modeButtons && ctx.modeButtons.length > 0) {
      ctx.modeButtons.forEach((button) => {
        button.addEventListener("click", () => {
          const nextMode = normalizeMode(button.dataset.modeChoice);
          if (nextMode === ctx.activeMode) {
            return;
          }
          ctx.activeMode = nextMode;
          updateContextualVisibility(ctx);
          handleFormChange(ctx);
        });
      });
    }
    if (ctx.sidebarToggleButton) {
      ctx.sidebarToggleButton.addEventListener("click", () => {
        updateSidebarState(ctx, !ctx.sidebarCollapsed);
      });
    }
  }

  function createContext(variant) {
    return {
      variant,
      storageKey: getConfigStorageKey(variant),
      latestCommandText: DEFAULT_CLASSIC_COMMAND,
      latestCommandMeta: null,
      modelCatalog: getWindowModelCatalog(),
      priceCatalog: getWindowPricingCatalog(),
      sampledEstimateRows: null,
      sampledEstimateSource: "",
      isInitializing: true,
      activeMode: "run",
      sidebarCollapsed: true,
      fieldHelpDismissListenersBound: false,
      form: document.getElementById("config-form"),
      commandPanel: document.querySelector(".command-panel"),
      commandOutput: document.getElementById("command-output"),
      promptTokenEstimate: document.getElementById("prompt-token-estimate"),
      promptTokenEstimateStatus: document.getElementById("prompt-token-estimate-status"),
      promptPreviewSystem: document.getElementById("prompt-preview-system"),
      promptPreviewUser: document.getElementById("prompt-preview-user"),
      promptPreviewMeta: document.getElementById("prompt-preview-meta"),
      estimateCsvFileInput: document.getElementById("estimate-csv-file"),
      sampleCsvButton: document.getElementById("sample-csv-button"),
      copyButton: document.getElementById("copy-button"),
      copyButtonDefaultText:
        document.getElementById("copy-button")?.textContent?.trim() || "Copy",
      resetButton: document.getElementById("reset-button"),
      providerSelect: document.getElementById("provider"),
      vertexAuthOptions: document.getElementById("vertex-auth-options"),
      apiKeyVarInput: document.getElementById("api_key_var"),
      apiBaseVarInput: document.getElementById("api_base_var"),
      refreshModelsButton: document.getElementById("refresh-models-button"),
      refreshModelsStatus: document.getElementById("refresh-models-status"),
      serviceTierInput: document.getElementById("service_tier"),
      modelInput: document.getElementById("model"),
      modelClearButton: document.getElementById("model-clear-button"),
      modelList: document.getElementById("model-options"),
      modelCatalogMeta: document.getElementById("model-catalog-meta"),
      createGeminiCacheCheckbox: document.getElementById("create_gemini_cache"),
      logprobsInput: document.getElementById("logprobs"),
      inputPathInput: document.getElementById("input_path"),
      outputPathInput: document.getElementById("output_path"),
      outputPathModeBadge: document.getElementById("output-path-mode-badge"),
      outputPathModeHint: document.getElementById("output-path-mode-hint"),
      modeButtons: Array.from(document.querySelectorAll("[data-mode-choice]")),
      sidebarToggleButton: document.getElementById("sidebar-toggle-button"),
      sidebarModeLabel: document.getElementById("sidebar-mode-label"),
      sidebarFlagCount: document.getElementById("sidebar-flag-count"),
      modeLabelElements: Array.from(document.querySelectorAll("[data-current-mode-label]")),
      cliFlagReference: document.getElementById("cli-flag-reference"),
    };
  }

  function initClassicPage(ctx) {
    syncProvidersFromCatalog(ctx);
    loadConfig(ctx);
    renderCliFlagReference(ctx);
    applyInputHoverHelp(ctx);
    updatePlaceholdersForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
    updateModelOptionsForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
    updateContextualVisibility(ctx);
    bindCommonListeners(ctx);
    ctx.isInitializing = false;
    updateContextualVisibility(ctx);
    handleFormChange(ctx);
  }

  function initPreviewPage(ctx) {
    syncProvidersFromCatalog(ctx);
    loadConfig(ctx);
    renderCliFlagReference(ctx);
    ctx.sidebarCollapsed = loadSidebarState(ctx);
    applyInputHoverHelp(ctx);
    updatePlaceholdersForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
    updateModelOptionsForProvider(ctx, ctx.providerSelect?.value || defaultValues.provider);
    updateContextualVisibility(ctx);
    updateSidebarState(ctx, ctx.sidebarCollapsed, false);
    bindCommonListeners(ctx);
    bindPreviewModeListeners(ctx);
    ctx.isInitializing = false;
    updateContextualVisibility(ctx);
    handleFormChange(ctx);
  }

  function initializeGui() {
    const form = document.getElementById("config-form");
    if (!form) {
      return;
    }
    const variant =
      document.body && document.body.dataset && document.body.dataset.guiVariant
        ? document.body.dataset.guiVariant
        : CLASSIC_VARIANT;
    const ctx = createContext(variant);
    if (isModeFirstVariant(variant)) {
      initPreviewPage(ctx);
    } else {
      initClassicPage(ctx);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeGui);
  } else {
    initializeGui();
  }
})();
