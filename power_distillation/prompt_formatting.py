"""Prompt formatting helpers shared by training and evaluation."""

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MATH_USER_TEMPLATE = (
    "Can you solve the following math problem? {question} "
    "Please reason step by step, and put your final answer within \\boxed{{}}. "
)
GENERIC_TEMPLATE = (
    "Can you complete the following task? {question} "
    "Please reason step by step, and clearly indicate the final answer."
)


def extract_prompt_text(prompt_item):
    if isinstance(prompt_item, str):
        return prompt_item
    if isinstance(prompt_item, dict) and isinstance(prompt_item.get("prompt"), str):
        return prompt_item["prompt"]
    raise ValueError("Prompt must be a string or a dict with a string 'prompt' field.")


def format_generation_prompt(tokenizer, prompt_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    prompt_text = extract_prompt_text(prompt_text)
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_user_assistant_text(tokenizer, prompt_text, completion_text):
    prompt_text = extract_prompt_text(prompt_text)
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": completion_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_user_prompt_with_generation_marker(tokenizer, prompt_text):
    prompt_text = extract_prompt_text(prompt_text)
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_math_user_prompt(question):
    return MATH_USER_TEMPLATE.format(question=extract_prompt_text(question))


def build_user_prompt(question):
    return GENERIC_TEMPLATE.format(question=extract_prompt_text(question))


def format_math_generation_prompt(tokenizer, question, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return format_generation_prompt(
        tokenizer,
        build_math_user_prompt(question),
        system_prompt=system_prompt,
    )


def format_general_generation_prompt(tokenizer, question, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return format_generation_prompt(
        tokenizer,
        build_user_prompt(question),
        system_prompt=system_prompt,
    )

