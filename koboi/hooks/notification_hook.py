"""koboi/hooks/notification_hook.py -- Desktop notifications on agent events.

Subscribes to configurable hook events and sends desktop notifications
with optional sound alerts. Primarily useful in --no-tui mode where the
Textual app's built-in notification handlers are not active.
"""

from __future__ import annotations

from koboi.hooks.chain import Hook, HookContext, HookEvent


# Default event-to-message mapping
_EVENT_MESSAGES: dict[HookEvent, tuple[str, str]] = {
    HookEvent.POST_OUTPUT: ("Koboi Agent", "Response complete"),
    HookEvent.DOOM_LOOP_DETECTED: ("Koboi Agent", "Doom loop detected"),
    HookEvent.AGENT_COMPLETED: ("Koboi Agent", "Sub-agent completed"),
    HookEvent.SESSION_END: ("Koboi Agent", "Session ended"),
}


class NotificationHook(Hook):
    """Send desktop notifications on selected lifecycle events.

    Config:
        events: list of HookEvent values to notify on (default: POST_OUTPUT)
        sound: whether to play a sound with each notification
        sound_name: macOS sound name (default: "Ping")
    """

    def __init__(
        self,
        events: list[HookEvent] | None = None,
        sound: bool = False,
        sound_name: str = "Ping",
    ) -> None:
        self._events = events or [HookEvent.POST_OUTPUT]
        self._sound = sound
        self._sound_name = sound_name

    def handles(self) -> list[HookEvent]:
        return list(self._events)

    async def execute(self, ctx: HookContext) -> HookContext:
        from koboi.notifications import notify, play_sound

        entry = _EVENT_MESSAGES.get(ctx.event)
        if entry:
            title, message = entry
        else:
            title, message = "Koboi Agent", ctx.event.value

        # Enrich message with context
        if ctx.event == HookEvent.POST_OUTPUT and ctx.llm_response:
            content = getattr(ctx.llm_response, "content", None)
            if content:
                preview = content[:100].replace("\n", " ")
                message = f"Response: {preview}..."
        elif ctx.event == HookEvent.DOOM_LOOP_DETECTED:
            message = "Repeated action pattern detected"
        elif ctx.event == HookEvent.AGENT_COMPLETED:
            name = ctx.metadata.get("agent_name", "unknown")
            message = f"Agent '{name}' completed"

        notify(title, message, sound=self._sound)
        if self._sound:
            play_sound(self._sound_name)

        return ctx
