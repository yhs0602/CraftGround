"""CraftGround IPC protocol handshake (docs/26_2_MigrationPlan.md item (f)).

Bumped whenever InitialEnvironmentMessage/ObservationSpaceMessage/ActionSpaceMessageV2 change in a
way that breaks wire compatibility with older Java/Python builds. Must match shared-java's
com.kyhsgeekcode.minecraftenv.CRAFTGROUND_PROTOCOL_VERSION.
"""

from .proto.initial_environment_pb2 import HandshakeAck

CRAFTGROUND_PROTOCOL_VERSION = 1


class ProtocolVersionMismatchError(RuntimeError):
    """Raised when the Java side's HandshakeAck reports an incompatible protocol_version."""


def validate_handshake_ack(ack: HandshakeAck) -> None:
    if ack.protocol_version != CRAFTGROUND_PROTOCOL_VERSION:
        raise ProtocolVersionMismatchError(
            "CraftGround protocol mismatch: this Python package speaks protocol "
            f"{CRAFTGROUND_PROTOCOL_VERSION}, but the connected Java side (Minecraft "
            f"{ack.minecraft_version}, render backend '{ack.render_backend}') speaks protocol "
            f"{ack.protocol_version}. Reinstall matching craftground and "
            "craftground-runtime-mc121/mc262 versions."
        )
