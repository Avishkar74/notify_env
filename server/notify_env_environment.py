"""Compatibility wrapper for the new notification environment implementation."""

from .environment import NotificationEnvironment


class NotifyEnvironment(NotificationEnvironment):
    """Backwards-compatible class name used by existing server wiring."""

    pass
