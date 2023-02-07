__all__ = ["RequiredWithOption", "MutexOption"]

import click

ERROR_REQUIRED_WITH_KWARG = (
    "RequiredWithOption requires 'required_with' keyword argument."
)
ERROR_MUTEX_KWARG = "MutexOption requires 'mutex' keyword argument."
USAGE_ERROR_MESSAGE_REQUIRED_WITH = (
    "Option '{name}' is required with option(s) {options}."
)
USAGE_ERROR_MESSAGE_MUTEX_TOO_FEW = (
    "Please define at least one of the following options {}."
)
USAGE_ERROR_MESSAGE_MUTEX_TOO_MANY = "Option '{}' is mutually exclusive with {}."


class RequiredWithOption(click.Option):
    def __init__(self, *args, **kwargs) -> None:
        if "required_with" not in kwargs:
            raise ValueError(ERROR_REQUIRED_WITH_KWARG)
        self.required_with = set(kwargs.pop("required_with"))
        kwargs["help"] = "{}  [required with: {}]".format(
            kwargs.get("help", ""), ", ".join(list(self.required_with))
        )
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):  # noqa
        # Check if required if arguments have been passed.
        if self.name not in opts and len(self.required_with.intersection(opts)) > 0:
            raise click.UsageError(
                USAGE_ERROR_MESSAGE_REQUIRED_WITH.format(
                    name=self.name, options=list(self.required_with)
                )
            )
        return super().handle_parse_result(ctx, opts, args)


class MutexOption(click.Option):
    def __init__(self, *args, **kwargs) -> None:
        if "mutex" not in kwargs:
            raise ValueError(ERROR_MUTEX_KWARG)
        self.require_one = False
        if "required" in kwargs:
            self.require_one = kwargs.pop("required")
        self.mutex = set(kwargs.pop("mutex"))
        kwargs["help"] = "{}  [mutex: {}]".format(
            kwargs.get("help", ""), ", ".join(list(self.mutex))
        )
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):  # noqa
        # Remove name from mutex.
        if self.name in self.mutex:
            self.mutex.remove(self.name)
        # Check if neither options have been passed.
        if (
            self.require_one
            and len(self.mutex.intersection(opts)) == 0
            and self.name not in opts
        ):
            raise click.UsageError(
                USAGE_ERROR_MESSAGE_MUTEX_TOO_FEW.format([self.name, *list(self.mutex)])
            )
        # Check mutual exclusivity.
        if len(self.mutex.intersection(opts)) > 0 and self.name in opts:
            raise click.UsageError(
                USAGE_ERROR_MESSAGE_MUTEX_TOO_MANY.format(self.name, self.mutex)
            )
        return super().handle_parse_result(ctx, opts, args)
