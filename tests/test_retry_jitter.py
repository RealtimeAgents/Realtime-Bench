"""The shared retry policy's jitter (entity/model.py TransportRetryPolicy.delay_s)."""

from realtime_framework.entity.model import TRANSIENT, TRANSPORT_RETRY, TransportFailure


def test_keys_one_character_apart_are_jittered_apart():
    """The jitter is a hash of the request unit's key: the readying keys of consecutive frames,
    which differ in one character, wait lengths spread over the jitter range, not ones a few
    thousandths of a second apart."""
    failure = TransportFailure(TRANSIENT)
    delays = [TRANSPORT_RETRY.delay_s(1, f"episode:f{k}.frame:ready", failure) for k in range(10)]
    assert all(0.75 <= delay <= 1.25 for delay in delays)
    assert max(delays) - min(delays) > 0.25
