"""Harbor integration. Job files name the agent and environment classes by this package's path."""


def __getattr__(name):
    if name in ("ApptainerEnvironment", "RealtimeBenchAgent"):
        from realtime_framework.evaluation.harbor import integration

        return getattr(integration, name)
    raise AttributeError(name)
