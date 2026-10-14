"""Select the model decision strategy independently of provider transport."""

from realtime_framework.providers.registry import (
    adapter_capabilities,
    adapter_class,
    load_project_secrets,
)


def create_agent(config, log_dir=None):
    load_project_secrets()
    adapter_class(config.adapter_type)  # Resolve and validate the lazy driver target.
    if adapter_capabilities(config.adapter_type).action_channel == "choice":
        from realtime_framework.harnesses.jev.harness import JevHarness

        return JevHarness(config, log_dir=log_dir)
    from realtime_framework.harnesses.default.harness import DefaultHarness

    return DefaultHarness(config, log_dir=log_dir)
