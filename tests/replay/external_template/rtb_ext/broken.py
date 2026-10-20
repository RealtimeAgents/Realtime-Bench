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
    model_defaults: ClassVar[dict[str, dict]] = {}


class NoContextStatus(ExternalChatDriver):
    request_context_status = None


class NoCapabilities(ExternalChatDriver):
    capabilities = None


class NoDeployment(ExternalChatDriver):
    deployment = None


class NoModelDefaults(ExternalChatDriver):
    model_defaults = None


class ClassifierFlagOnly(ExternalChatDriver):
    classify_transport_error = None


class RenewalFlagOnly(ExternalChatDriver):
    capabilities = replace(ExternalChatDriver.capabilities, connection_renewal=True)


class OutputCapFlagOnly(ExternalChatDriver):
    capabilities = replace(ExternalChatDriver.capabilities, maps_output_cap=True)
