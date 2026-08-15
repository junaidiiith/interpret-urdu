import torch
import torch.nn as nn

from urdu_interp.causal import patch_hidden_state, resolve_decoder_layers, steer_hidden_state


class _IdentityLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states * 2  # a trivial, deterministic transform to prove the hook sees post-layer output


class _TupleLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states * 2, "some_other_return_value")


class _FakeModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)


def test_resolve_decoder_layers_plain_layout():
    model = _FakeModel([_IdentityLayer()])
    assert list(resolve_decoder_layers(model)) == list(model.model.layers)


def test_resolve_decoder_layers_raises_when_nothing_matches():
    class _Empty(nn.Module):
        pass

    try:
        resolve_decoder_layers(_Empty())
        assert False, "expected AttributeError"
    except AttributeError:
        pass


def test_patch_hidden_state_overwrites_target_position_only():
    model = _FakeModel([_IdentityLayer(), _IdentityLayer()])
    x = torch.ones(1, 3, 4)
    replacement = torch.full((4,), 99.0)

    with patch_hidden_state(model, layer=1, position=1, replacement=replacement):
        out = model.model.layers[0](x)

    assert torch.equal(out[0, 0, :], torch.full((4,), 2.0))  # untouched position, sees the real layer output
    assert torch.equal(out[0, 1, :], replacement)  # patched position
    assert torch.equal(out[0, 2, :], torch.full((4,), 2.0))


def test_patch_hidden_state_handles_tuple_outputs():
    model = _FakeModel([_TupleLayer()])
    x = torch.ones(1, 2, 4)
    replacement = torch.full((4,), 7.0)

    with patch_hidden_state(model, layer=1, position=-1, replacement=replacement):
        out = model.model.layers[0](x)

    assert isinstance(out, tuple)
    assert torch.equal(out[0][0, -1, :], replacement)
    assert out[1] == "some_other_return_value"


def test_patch_hidden_state_removes_hook_after_context():
    model = _FakeModel([_IdentityLayer()])
    x = torch.ones(1, 2, 4)
    with patch_hidden_state(model, layer=1, position=0, replacement=torch.zeros(4)):
        pass
    out = model.model.layers[0](x)
    assert torch.equal(out, x * 2)  # no lingering hook


def test_steer_hidden_state_adds_to_every_position():
    model = _FakeModel([_IdentityLayer()])
    x = torch.ones(1, 2, 4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])

    with steer_hidden_state(model, layer=1, direction=direction, alpha=10.0):
        out = model.model.layers[0](x)

    expected = x * 2 + 10.0 * direction
    assert torch.allclose(out, expected)
