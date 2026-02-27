# Inference

## AceRAG-Llama2-7b

### Parameter Description

- `prompts` (`list`): A list containing user instructions, containing `CONTEXT_TAG` placeholders.
- `contexts` (`list`): A list containing retrieved context for each prompt.
- `comp_ratio`(`int`): Compression ratio that controls how aggressively the context is compressed. Higher values mean stronger compression. The model was trained with compression ratios of 1, 2, 4, and 8, and generalizes to ratios ranging from 1 to 64.
- `task_instruction`(`str`): Guides the compression model on how to compress the context for a specific task.
  * QA: "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n"
  * Summary: "Summarize the given documents."
  * In-context learning: " Compress the few-shot examples, retaining key patterns and diversity while removing redundancy."
  * Reasoning: "Compress the context to the core facts and relationships that are pivotal for solving the reasoning task."

### Example

Here is an example of 8× context compression using uniform down-sampling.

```python
from transformers import AutoModel, AutoTokenizer

CONTEXT_TAG = "[CONTEXT_RmehNsY1]"
INPUT_TAG = "[INPUT_RmehNsY1]"

tokenizer = AutoTokenizer.from_pretrained("wcyno23/AceRAG-Llama2-7b", trust_remote_code=True)
model = AutoModel.from_pretrained("wcyno23/AceRAG-Llama2-7b", trust_remote_code=True)
model = model.to('cuda')

qa_template = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:"
questions = ["What happened with Will Smith at the 2022 Oscars?"]
prompts = [qa_template.replace(INPUT_TAG, question) for question in questions]
contexts = ["""During the 94th Academy Awards on March 27, 2022, actor Will Smith walked onstage and slapped comedian Chris Rock after Rock made a joke about Jada Pinkett Smith's shaved head (due to her alopecia). Smith then returned to his seat and yelled "Keep my wife's name out your fucking mouth!" The moment went viral instantly, shocked the audience, and dominated headlines worldwide. Smith later won Best Actor for King Richard that night, gave an emotional acceptance speech apologizing (but not to Rock), and was banned from the Academy for 10 years. It sparked massive debates on celebrity behavior, comedy boundaries, alopecia awareness, and live TV moments."""]

compression_instruction = "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n"

inputs = model.compression_rate_adapter.uniform_allocation(
    prompts, 
    contexts, 
    tokenizer=tokenizer, 
    comp_ratio=8,
    task_instruction=compression_instruction,
)

inputs = model._move_to_device(inputs)
outputs = model.generate(**inputs)
outputs = outputs[:, inputs["input_ids"].shape[1]:] # return only generated
print(f"Question: {questions[0]}")
print(f"Answers: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

### Advanced Example

During down-sampling, model performance can be further improved by leveraging compression ratio adaptation. 

For example, tokens can be divided into two priority levels: high-priority tokens are assigned lower compression ratios, while low-priority tokens are assigned higher compression ratios. 

Specifically, AceRAG employs an embedding model trained for token-level retrieval to estimate token priorities. **Note:** this evaluation introduces additional computational overhead.

- `queries` (`list`): A list of questions to be used as input to the embedding model.

- `context_proportion` (`float`): The proportion of high-priority tokens.

- `low_comp_ratio` (`int`): Compression ratios assigned to high-priority tokens.

**Note**: Ensure that the values of `context_proportion` and `low_comp_ratio` are valid, so that the compression ratios for the remaining context can be computed correctly and the overall compression ratio is achievable.

```python
inputs = model.compression_rate_adapter.token_level_adaptation(
    prompts,
    contexts,
    tokenizer=tokenizer,
    queries=[question + " </s>" for question in questions],
    comp_ratio=8,
    task_instruction=compression_instruction,
    context_proportion=0.0625,
    low_comp_ratio=1,
)

inputs = model._move_to_device(inputs, model.device)
outputs = model.generate(**inputs)
outputs = outputs[:, inputs["input_ids"].shape[1]:] # return only generated
print(f"Question: {questions[0]}")
print(f"Answers: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

## AceRAG-Qwen3-8b

### Parameter Description

- `prompts` (`list`): A list containing user instructions, containing `CONTEXT_TAG` placeholders.
- `contexts` (`list`): A list containing retrieved context for each prompt.
- `comp_ratio`(`int`): Compression ratio that controls how aggressively the context is compressed. Higher values mean stronger compression. The model was trained with compression ratios of 1, 2, 4, and 8, and generalizes to ratios ranging from 1 to 64.
- `task_instruction`(`str`): Guides the compression model on how to compress the context for a specific task.
  * QA: "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n"
  * Summary: "Summarize the given documents."
  * In-context learning: " Compress the few-shot examples, retaining key patterns and diversity while removing redundancy."
  * Reasoning: "Compress the context to the core facts and relationships that are pivotal for solving the reasoning task."

### Example

Here is an example of 8× context compression using uniform down-sampling.

```python
from transformers import AutoModel, AutoTokenizer

CONTEXT_TAG = "[CONTEXT_RmehNsY1]"
INPUT_TAG = "[INPUT_RmehNsY1]"

tokenizer = AutoTokenizer.from_pretrained("wcyno23/AceRAG-Qwen3-8b", trust_remote_code=True)
model = AutoModel.from_pretrained("wcyno23/AceRAG-Qwen3-8b", trust_remote_code=True)
model = model.to('cuda')

qa_template = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:"
questions = ["What happened with Will Smith at the 2022 Oscars?"]
prompts = [qa_template.replace(INPUT_TAG, question) for question in questions]
contexts = ["""During the 94th Academy Awards on March 27, 2022, actor Will Smith walked onstage and slapped comedian Chris Rock after Rock made a joke about Jada Pinkett Smith's shaved head (due to her alopecia). Smith then returned to his seat and yelled "Keep my wife's name out your fucking mouth!" The moment went viral instantly, shocked the audience, and dominated headlines worldwide. Smith later won Best Actor for King Richard that night, gave an emotional acceptance speech apologizing (but not to Rock), and was banned from the Academy for 10 years. It sparked massive debates on celebrity behavior, comedy boundaries, alopecia awareness, and live TV moments."""]

compression_instruction = "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n"

inputs = model.compression_rate_adapter.uniform_allocation(
    prompts, 
    contexts, 
    tokenizer=tokenizer, 
    comp_ratio=8,
    task_instruction=compression_instruction,
)

inputs = model._move_to_device(inputs)
outputs = model.generate(**inputs)
outputs = outputs[:, inputs["input_ids"].shape[1]:] # return only generated
print(f"Question: {questions[0]}")
print(f"Answers: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

### Advanced Example

During down-sampling, model performance can be further improved by leveraging compression ratio adaptation. 

For example, tokens can be divided into two priority levels: high-priority tokens are assigned lower compression ratios, while low-priority tokens are assigned higher compression ratios. 

Specifically, AceRAG employs an embedding model trained for token-level retrieval to estimate token priorities. **Note:** this evaluation introduces additional computational overhead.

- `queries` (`list`): A list of questions to be used as input to the embedding model.

- `context_proportion` (`float`): The proportion of high-priority tokens.

- `low_comp_ratio` (`int`): Compression ratios assigned to high-priority tokens.

**Note**: Ensure that the values of `context_proportion` and `low_comp_ratio` are valid, so that the compression ratios for the remaining context can be computed correctly and the overall compression ratio is achievable.

```python
inputs = model.compression_rate_adapter.token_level_adaptation(
    prompts,
    contexts,
    tokenizer=tokenizer,
    queries=[question + "<endoftext>" for question in questions],
    comp_ratio=8,
    task_instruction=compression_instruction,
    context_proportion=0.0625,
    low_comp_ratio=1,
)

inputs = model._move_to_device(inputs, model.device)
outputs = model.generate(**inputs)
outputs = outputs[:, inputs["input_ids"].shape[1]:] # return only generated
print(f"Question: {questions[0]}")
print(f"Answers: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```