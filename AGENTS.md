# LLM Linguistic Classification Benchmark Agent

## Agent Overview
This agent is a Python-based benchmarking tool designed to evaluate large language models (LLMs) on a linguistic classification task. Each input consists of a text excerpt containing a specific **node word**, and the agent prompts an LLM (via the OpenAI API) to classify that node word according to a given linguistic criterion (e.g. a syntactic category like part-of-speech, or a semantic category). The agent automates the end-to-end process: loading a dataset of labeled examples, querying the LLM for each example, collecting the model’s predicted label along with an explanation and a self-reported confidence, and finally comparing these predictions against the ground truth labels to measure performance. This provides a reproducible way to benchmark an LLM’s accuracy on specialized linguistic classification problems.

... [truncated for brevity in this context]


## Additional Features

### Web-Based Configuration Interface

To enhance usability, the agent includes an HTML/JavaScript-based graphical user interface (GUI) for configuring the Python script. This configuration tool generates a command-line string that reflects all chosen options, making it easier for users to run the benchmarking script with their desired parameters.

### GUI Capabilities

The HTML/JavaScript GUI allows users to set the following parameters:

1. **Model Settings:**
   - **Model Name** (e.g., `gpt-4`, `gpt-3.5-turbo`)
   - **Temperature** (controls randomness; typically set to `0` for deterministic classification tasks)
   - **Top-p** (nucleus sampling; usually set to `1.0` to allow full token probability mass)
   - **Top-k** (optional: limits output to top-k tokens)

2. **Prompt Behavior:**
   - **Chain of Thought Toggle**: Enables or disables a structured reasoning style in the prompt. When enabled, the prompt encourages the model to explain its reasoning step-by-step before choosing a label.
   - **System Prompt**: Allows the user to set a custom system prompt (e.g., “You are a linguistic classifier that excels at semantic disambiguation.”). This will be inserted in the `system` role field for chat models.

3. **Input & Output:**
   - **Path to Input CSV File**
   - **Path to Ground Truth CSV File** (if separate)
   - **Output File Path** (for results)
   - **Enable Calibration Plot**: Checkbox for toggling calibration output

### Output

The GUI dynamically generates a CLI command (e.g., for `benchmark_agent.py`) such as:

```
python benchmark_agent.py --input data.csv --labels labels.csv --output results.csv \
  --model gpt-4 --temperature 0.0 --top_p 1.0 --top_k 5 \
  --system_prompt "You are a linguistic classifier..." \
  --enable_cot --calibration
```

This command can be copied and executed directly in a terminal to run the Python agent with the selected settings. The GUI is designed to run entirely client-side in the browser and does not require server-side components.
