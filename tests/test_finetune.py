import math

from urdu_interp.corpora import AlignedTriple
from urdu_interp.finetune import chunk_texts, eval_perplexity_detailed, rekhta_training_texts


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class _FakeLossOut:
    def __init__(self, loss):
        self.loss = loss


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeCausalModel:
    """Returns a fixed loss per call, in order -- enough to test
    eval_perplexity_detailed's block-loss/CI bookkeeping without a real
    model (which eval_perplexity/eval_perplexity_detailed otherwise need)."""

    def __init__(self, losses: list[float]):
        self._losses = iter(losses)

    def eval(self):
        pass

    def __call__(self, input_ids=None, labels=None):
        return _FakeLossOut(_FakeScalar(next(self._losses)))


def test_rekhta_training_texts_extracts_requested_language_field():
    triples = [
        AlignedTriple(id="a", source="s", ur="ur-text-a", hi="hi-text-a"),
        AlignedTriple(id="b", source="s", ur="ur-text-b", hi=None),
    ]
    texts = rekhta_training_texts(triples, lang="ur")
    assert texts == ["ur-text-a", "ur-text-b"]

    hi_texts = rekhta_training_texts(triples, lang="hi")
    assert hi_texts == ["hi-text-a"]


def test_chunk_texts_splits_concatenated_stream_into_fixed_blocks():
    tok = FakeTokenizer()
    blocks = chunk_texts(["ab", "cd"], tok, block_size=3)
    # ids: [97, 98, 999] + [99, 100, 999] -> two blocks of 3
    assert blocks == [[97, 98, 999], [99, 100, 999]]


def test_chunk_texts_drops_incomplete_trailing_block():
    tok = FakeTokenizer()
    blocks = chunk_texts(["abcdefghi"], tok, block_size=3)
    # ids: 9 chars + 1 eos = 10 -> 3 full blocks (9 ids), last id dropped
    assert len(blocks) == 3
    assert sum(len(b) for b in blocks) == 9


def test_chunk_texts_empty_when_shorter_than_block_size():
    tok = FakeTokenizer()
    assert chunk_texts(["a"], tok, block_size=10) == []


def test_eval_perplexity_detailed_reports_block_losses_and_matches_perplexity():
    tok = FakeTokenizer()
    # "ab" + eos -> block of size 3 (ids: 97, 98, 999); "cd" + eos -> another
    # block of size 3 -- two blocks, two fake losses.
    model = FakeCausalModel(losses=[1.0, 3.0])
    result = eval_perplexity_detailed(model, tok, ["ab", "cd"], block_size=3, device="cpu", n_boot=200, seed=0)
    assert result.n_blocks == 2
    assert result.block_losses == [1.0, 3.0]
    assert math.isclose(result.perplexity, math.exp(2.0), rel_tol=1e-6)  # mean loss = 2.0


def test_eval_perplexity_detailed_ci_widens_with_block_variance():
    tok = FakeTokenizer()
    texts = ["ab", "cd", "ef", "gh"]
    tight_model = FakeCausalModel(losses=[2.0, 2.0, 2.0, 2.0])
    wide_model = FakeCausalModel(losses=[0.5, 3.5, 0.5, 3.5])

    tight = eval_perplexity_detailed(tight_model, tok, texts, block_size=3, device="cpu", n_boot=500, seed=0)
    wide = eval_perplexity_detailed(wide_model, tok, texts, block_size=3, device="cpu", n_boot=500, seed=0)

    assert (tight.ci_high - tight.ci_low) < (wide.ci_high - wide.ci_low)


def test_eval_perplexity_detailed_raises_on_no_blocks():
    tok = FakeTokenizer()
    model = FakeCausalModel(losses=[])
    try:
        eval_perplexity_detailed(model, tok, ["a"], block_size=10, device="cpu")
        assert False, "expected ValueError"
    except ValueError:
        pass
