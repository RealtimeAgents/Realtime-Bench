"""External drivers with one declaration missing or wrong; loading their YAML is refused."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from rtb_ext.chat import ExternalChatDriver


def not_a_class(config):
    return ExternalChatDriver(config)


class NotAnEncoder:
    capabilities = ExternalChatDriver.capabilities
    deployment = ExternalChatDriver.deployment
    compat = ExternalChatDriver.compat
    model_defaults: ClassVar[dict[str, dict]] = {}


class NoClock(ExternalChatDriver):
    """Written to the contract before drivers took the episode clock."""

    def __init__(self, config):
        super().__init__(config)


class NoContextStatus(ExternalChatDriver):
    request_context_status = None


class NoCapabilities(ExternalChatDriver):
    capabilities = None


class NoDeployment(ExternalChatDriver):
    deployment = None


class NoCompat(ExternalChatDriver):
    compat = None


class NoModelDefaults(ExternalChatDriver):
    model_defaults = None


class ClassifierFlagOnly(ExternalChatDriver):
    classify_transport_error = None


class RenewalFlagOnly(ExternalChatDriver):
    capabilities = replace(ExternalChatDriver.capabilities, connection_renewal=True)


class OutputCapFlagOnly(ExternalChatDriver):
    capabilities = replace(ExternalChatDriver.capabilities, maps_output_cap=True)


class StreamingNoSwitch(ExternalChatDriver):
    capabilities = replace(
        ExternalChatDriver.capabilities, interaction_modes=frozenset({"turn_based", "streaming"})
    )
    streaming_settings: ClassVar[dict] = {}


class StreamingNoSettings(StreamingNoSwitch):
    streaming_settings = None

    def __init__(self, config, *, clock=None, streaming=False):
        super().__init__(config, clock=clock)


class StreamingBadSettings(StreamingNoSettings):
    streaming_settings: ClassVar[dict] = {"compression_trigger_tokens": 0}


class StreamingNoPush(StreamingNoSettings):
    streaming_settings: ClassVar[dict] = {}


class StreamingNoReply(StreamingNoPush):
    async def push_frame(self, observation, history):
        raise NotImplementedError


class StreamingNoClassifier(StreamingNoReply):
    """Declares no classifier, which the native-streaming agent calls on every failure."""

    capabilities = replace(StreamingNoReply.capabilities, classifies_transport_errors=False)
    classify_transport_error = None

    async def request_reply(self, request):
        raise NotImplementedError


class ChoiceNoDecide(ExternalChatDriver):
    capabilities = replace(ExternalChatDriver.capabilities, action_channel="choice")


class ChoiceNoClassifier(ChoiceNoDecide):
    async def decide(self, state, question):
        raise NotImplementedError
