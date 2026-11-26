const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Load the actual command builders without starting the browser UI.
const source = fs.readFileSync(path.join(__dirname, "..", "config_gui_shared.js"), "utf8");
const context = vm.createContext({
  document: { readyState: "loading", addEventListener() {}, getElementById() { return null; } },
  window: { btoa: (text) => Buffer.from(text, "binary").toString("base64") },
});
vm.runInContext(source.replace(/\}\)\(\);\s*$/, `
  globalThis.builders = { buildCommandClassic, buildCommandPreview };
})();`), context);
const { buildCommandClassic, buildCommandPreview } = context.builders;
const prompt = "Classify grammatical number. Use singular or plural.";
const legacy = "You are a linguistic classifier that excels at semantic disambiguation.";
const data = (systemPrompt) => new Map([
  ["provider", "openai"], ["model", "test-model"], ["input_path", "example_input.csv"],
  ["include_explanations", true], ["system_prompt", systemPrompt],
]);

test("run and validator commands reject absent, blank and former default prompts", () => {
  for (const value of [undefined, "", " \n\t", legacy]) {
    for (const activeMode of ["run", "validator"]) {
      assert.throws(() => buildCommandPreview({ activeMode }, data(value)), /task-specific system prompt/);
    }
    assert.throws(() => buildCommandClassic(data(value)), /task-specific system prompt/);
  }
});

test("both builders emit explicit plain and multiline prompt flags", () => {
  for (const build of [buildCommandClassic, (values) => buildCommandPreview({ activeMode: "run" }, values)]) {
    const plain = build(data(prompt));
    assert.ok(plain.flags.includes("--system_prompt"));
    const multilinePrompt = prompt + "\nUse the context.";
    const multiline = build(data(multilinePrompt));
    const index = multiline.args.indexOf("--system_prompt_b64");
    assert.ok(index >= 0);
    assert.equal(Buffer.from(multiline.args[index + 1], "base64").toString("utf8"), multilinePrompt);
  }
});

test("metrics and resume modes can generate commands without a prompt", () => {
  for (const activeMode of ["metrics", "resume"]) {
    const result = buildCommandPreview({ activeMode }, data(""));
    assert.ok(result.flags.includes(activeMode === "metrics" ? "--metrics_only" : "--resume"));
    assert.ok(!result.flags.includes("--system_prompt"));
  }
  const metrics = data("");
  metrics.set("metrics_only", true);
  assert.ok(buildCommandClassic(metrics).flags.includes("--metrics_only"));
});
