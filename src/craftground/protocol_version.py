"""CraftGround IPC protocol handshake (docs/26_2_MigrationPlan.md item (f)).

Bumped whenever InitialEnvironmentMessage/ObservationSpaceMessage/ActionSpaceMessageV2 change in a
way that breaks wire compatibility with older Java/Python builds. Must match shared-java's
com.kyhsgeekcode.minecraftenv.CRAFTGROUND_PROTOCOL_VERSION.
"""

from typing import Optional

from .proto.initial_environment_pb2 import HandshakeAck
from .screen_encoding_modes import ScreenEncodingMode

CRAFTGROUND_PROTOCOL_VERSION = 1

# Encoding modes that require the "zerocopy" capability Java's sendHandshakeAck only sets when
# the resolved capture backend actually has a working mach-port/CUDA-handle path (see
# EnvironmentInitializer.checkRenderBackend/sendHandshakeAck). Silently accepting the ack
# otherwise would let a ZEROCOPY_TORCH/ZEROCOPY_JAX request pass the handshake against a
# non-zerocopy-capable backend (e.g. "vulkan-cpu-readback"), only to fail later when Java itself
# throws in checkRenderBackend or Python tries to open a mach port/CUDA handle that was never sent.
_ZEROCOPY_MODES = {
    ScreenEncodingMode.ZEROCOPY_TORCH.value,
    ScreenEncodingMode.ZEROCOPY_JAX.value,
}


class ProtocolVersionMismatchError(RuntimeError):
    """Raised when the Java side's HandshakeAck reports an incompatible protocol_version."""


class RenderBackendCapabilityError(RuntimeError):
    """Raised when the requested screen encoding mode isn't supported by the ack'd render backend."""


def validate_handshake_ack(
    ack: HandshakeAck, requested_screen_encoding_mode: Optional[int] = None
) -> None:
    if ack.protocol_version != CRAFTGROUND_PROTOCOL_VERSION:
        raise ProtocolVersionMismatchError(
            "CraftGround protocol mismatch: this Python package speaks protocol "
            f"{CRAFTGROUND_PROTOCOL_VERSION}, but the connected Java side (Minecraft "
            f"{ack.minecraft_version}, render backend '{ack.render_backend}') speaks protocol "
            f"{ack.protocol_version}. Reinstall matching craftground and "
            "craftground-runtime-mc121/mc262 versions."
        )
    if (
        requested_screen_encoding_mode in _ZEROCOPY_MODES
        and "zerocopy" not in ack.capabilities
    ):
        raise RenderBackendCapabilityError(
            "CraftGround requested a ZEROCOPY encoding mode, but the connected Java side "
            f"(Minecraft {ack.minecraft_version}, render backend '{ack.render_backend}') did not "
            f"report the 'zerocopy' capability (capabilities={list(ack.capabilities)}). Use RAW "
            "or PNG instead, or enable a supported zerocopy backend (see "
            "docs/26_2_vulkan_capture.md)."
        )
