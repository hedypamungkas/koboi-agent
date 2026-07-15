"""koboi/media/providers -- built-in image providers.

Imported by ``koboi.media`` so ``@register_image_provider`` decorators fire on import
(idempotent). Add new providers here (e.g. ``comfyui``) and append an import line.
"""

from __future__ import annotations

from koboi.media.providers import mock as _mock  # noqa: F401
from koboi.media.providers import surplus as _surplus  # noqa: F401
