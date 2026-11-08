"""The bench's `thinking: false`: Qwen3's soft switch on a step's client."""

from types import SimpleNamespace

import pytest

from tests.simuhome.bench.models import NO_THINK, load_variant, variant_from_dict, without_thinking


def _client(seen):
    async def create(**kwargs):
        seen.append(kwargs)
        return "ok"
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.asyncio
async def test_the_switch_goes_into_the_system_prompt():
    seen = []
    client = without_thinking(_client(seen))
    await client.chat.completions.create(model="qwen3:8b", messages=[
        {"role": "system", "content": "You segment requests."}, {"role": "user", "content": "hi"}])
    assert seen[0]["messages"][0] == {"role": "system", "content": f"You segment requests.\n{NO_THINK}"}
    assert seen[0]["messages"][1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_a_call_without_a_system_message_gets_one():
    seen = []
    await without_thinking(_client(seen)).chat.completions.create(
        messages=[{"role": "user", "content": "hi"}])
    assert seen[0]["messages"][0] == {"role": "system", "content": NO_THINK}


def test_the_variant_reads_and_labels_the_switch():
    variant = variant_from_dict({"segmentation": {"model": "qwen3:8b", "thinking": False},
                                 "structuring": {"model": "qwen3:8b"}}, "v")
    assert variant["segmentation"].thinking is False and variant["structuring"].thinking is None
    assert variant["segmentation"].label() == "qwen3:8b@no-think"
    assert variant["structuring"].label() == "qwen3:8b"
    assert variant.as_dict()["steps"]["segmentation"]["thinking"] is False


def test_the_fep_variant_switches_the_understanding_phases_only():
    variant = load_variant("tests/simuhome/bench/variants/fep.ALL-qwen3-8b-nothink.yaml")
    assert variant["segmentation"].thinking is False and variant["structuring"].thinking is False
    assert all(variant[s].thinking is None for s in ("planning.general", "profiling", "response"))
