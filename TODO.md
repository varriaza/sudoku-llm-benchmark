Done
- [x] separate prompt_tokens vs completion_tokens
- [x] Add bits_per_weight to benchmark and its results (only if this can be found programatically)
- [x] Add peak_total_memory_mb to the benchmark and its results
- [x] Add model_file_size_gb to the benchmark and its results
- [x] Add the backend used for the test (vllm, llamacpp, etc)
- [x] Make llama.cpp set temperature to 0.1 for consistency

To Do
- [x] break total_tokens,context_tokens_used into "num_input_tokens", "num_thinking_tokens", "num_output_tokens" (note, these are separate from thinking tokens), "total_response_tokens" (num_thinking_tokens + num_output_tokens) "total_tokens_used" (eg num_input_tokens + num_thinking_tokens + num_output_tokens)
- [x] Add mean, median and max tokens per second (taken from the backend) to the benchmark and its results
- [x] Add mean, median and max prompt processing token speed (taken from the backend) to the benchmark and its results
- [x] Add the timestamp a run was started and finished to the results csv
- [ ] Hide some of the run output info (in the terminal) as it isn't really needed
  - from "uv run sudoku-bench configs/sanity_check.yaml"
- [ ] Save full LLM thinking and response text to a file for debugging why answers are a certain way
- [ ] Figure out why many tests at the same difficulty have wildly different success rates (eg 0% correct vs 100%)
- [ ] Figure out why so many tests have such few "total_turns"
  - Maybe empasis is needed on how much the LLM can rely on the resutls checker for intermediate work checking?
  - Maybe tell the LLM to only update at max 5 numbers a turn?
  - Also maybe have a run mode where each new post-turn LLM run is with clean context
    - Measures thinking at best case senarios 
- [ ] Figure out why we have so many malformed single respnses
  - Probably related to a lack of total_turns and 0% vs 100% success rates 
- [ ] Double check the default settings for llama.cpp to make sure things like auto prune context or other silent things that would influence tests are not present
- [ ] Add RAM safeguards (weights+full KV) (or since I have to have input context value size, use that) to the start of the run
  - eg quickly fail if estimated RAM use is greater than GPU vRAM + half of system ram
  - Estimate RAM from model via file size calculations
  - Estimate RAM from KV via size at full context length


Srapped todo
- (scraped) Make vllm work 
- Investigate "(APIServer pid=153890) WARNING 03-29 10:21:46 [interface.py:472] Using 'pin_memory=False' as WSL is detected. This may slow down the performance."
- Make vllm set temperature to 0.1 for consistency
- Double check the default settings for vLLM to make sure things like auto prune context or other silent things that would influence tests are not present