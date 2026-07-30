package com.kyhsgeekcode.minecraftenv

/**
 * Bumped whenever InitialEnvironmentMessage/ObservationSpaceMessage/ActionSpaceMessageV2 change in
 * a way that breaks wire compatibility with older Java/Python builds. Exchanged once per session
 * via [MessageIO.writeHandshakeAck]/`HandshakeAck.protocol_version` so an incompatible pairing
 * fails fast on the Python side instead of hanging or misinterpreting bytes deep into a session
 * (docs/26_2_MigrationPlan.md item (f)). Must match Python's
 * `craftground.protocol_version.CRAFTGROUND_PROTOCOL_VERSION`.
 */
const val CRAFTGROUND_PROTOCOL_VERSION = 1
