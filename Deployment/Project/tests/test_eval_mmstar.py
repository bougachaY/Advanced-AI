from types import SimpleNamespace

from eval_mmstar import (
    build_mmstar_prompt,
    extract_answer_letter,
    normalize_mmstar_question,
)


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.add_generation_prompt = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        assert tokenize is False
        return [messages[0][0]["content"] + "<GEN>"]


def test_extract_answer_letter():
    assert extract_answer_letter("A") == "A"
    assert extract_answer_letter("Answer: B") == "B"
    assert extract_answer_letter("The answer is C.") == "C"
    assert extract_answer_letter("not sure") is None


def test_normalize_mmstar_question_adds_letter_instruction():
    question = normalize_mmstar_question("What is shown?\nOptions:\nA. Cat")
    assert "Choices" in question
    assert "Answer with the letter directly." in question


def test_build_mmstar_prompt_prepends_image_tokens_and_uses_chat_template():
    tokenizer = FakeTokenizer()
    cfg = SimpleNamespace(
        projector=SimpleNamespace(image_token_length=3),
        image_token="<|image|>",
    )

    prompt = build_mmstar_prompt("What is shown?", tokenizer, cfg)

    assert prompt.startswith("<|image|><|image|><|image|>")
    assert "Answer with the letter directly." in prompt
    assert tokenizer.add_generation_prompt is True
    assert tokenizer.messages[0][0]["role"] == "user"
