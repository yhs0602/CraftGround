# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: observation_space.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x17observation_space.proto"o\n\tItemStack\x12\x0e\n\x06raw_id\x18\x01 \x01(\x05\x12\x17\n\x0ftranslation_key\x18\x02 \x01(\t\x12\r\n\x05\x63ount\x18\x03 \x01(\x05\x12\x12\n\ndurability\x18\x04 \x01(\x05\x12\x16\n\x0emax_durability\x18\x05 \x01(\x05"E\n\tBlockInfo\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\t\n\x01z\x18\x03 \x01(\x05\x12\x17\n\x0ftranslation_key\x18\x04 \x01(\t"\x87\x01\n\nEntityInfo\x12\x13\n\x0bunique_name\x18\x01 \x01(\t\x12\x17\n\x0ftranslation_key\x18\x02 \x01(\t\x12\t\n\x01x\x18\x03 \x01(\x01\x12\t\n\x01y\x18\x04 \x01(\x01\x12\t\n\x01z\x18\x05 \x01(\x01\x12\x0b\n\x03yaw\x18\x06 \x01(\x01\x12\r\n\x05pitch\x18\x07 \x01(\x01\x12\x0e\n\x06health\x18\x08 \x01(\x01"\x99\x01\n\tHitResult\x12\x1d\n\x04type\x18\x01 \x01(\x0e\x32\x0f.HitResult.Type\x12 \n\x0ctarget_block\x18\x02 \x01(\x0b\x32\n.BlockInfo\x12"\n\rtarget_entity\x18\x03 \x01(\x0b\x32\x0b.EntityInfo"\'\n\x04Type\x12\x08\n\x04MISS\x10\x00\x12\t\n\x05\x42LOCK\x10\x01\x12\n\n\x06\x45NTITY\x10\x02"L\n\x0cStatusEffect\x12\x17\n\x0ftranslation_key\x18\x01 \x01(\t\x12\x10\n\x08\x64uration\x18\x02 \x01(\x05\x12\x11\n\tamplifier\x18\x03 \x01(\x05"Q\n\nSoundEntry\x12\x15\n\rtranslate_key\x18\x01 \x01(\t\x12\x0b\n\x03\x61ge\x18\x02 \x01(\x03\x12\t\n\x01x\x18\x03 \x01(\x01\x12\t\n\x01y\x18\x04 \x01(\x01\x12\t\n\x01z\x18\x05 \x01(\x01"7\n\x16\x45ntitiesWithinDistance\x12\x1d\n\x08\x65ntities\x18\x01 \x03(\x0b\x32\x0b.EntityInfo"\x80\x08\n\x17ObservationSpaceMessage\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\t\n\x01x\x18\x02 \x01(\x01\x12\t\n\x01y\x18\x03 \x01(\x01\x12\t\n\x01z\x18\x04 \x01(\x01\x12\x0b\n\x03yaw\x18\x05 \x01(\x01\x12\r\n\x05pitch\x18\x06 \x01(\x01\x12\x0e\n\x06health\x18\x07 \x01(\x01\x12\x12\n\nfood_level\x18\x08 \x01(\x01\x12\x18\n\x10saturation_level\x18\t \x01(\x01\x12\x0f\n\x07is_dead\x18\n \x01(\x08\x12\x1d\n\tinventory\x18\x0b \x03(\x0b\x32\n.ItemStack\x12"\n\x0eraycast_result\x18\x0c \x01(\x0b\x32\n.HitResult\x12$\n\x0fsound_subtitles\x18\r \x03(\x0b\x32\x0b.SoundEntry\x12%\n\x0estatus_effects\x18\x0e \x03(\x0b\x32\r.StatusEffect\x12I\n\x11killed_statistics\x18\x0f \x03(\x0b\x32..ObservationSpaceMessage.KilledStatisticsEntry\x12G\n\x10mined_statistics\x18\x10 \x03(\x0b\x32-.ObservationSpaceMessage.MinedStatisticsEntry\x12\x45\n\x0fmisc_statistics\x18\x11 \x03(\x0b\x32,.ObservationSpaceMessage.MiscStatisticsEntry\x12%\n\x10visible_entities\x18\x12 \x03(\x0b\x32\x0b.EntityInfo\x12O\n\x14surrounding_entities\x18\x13 \x03(\x0b\x32\x31.ObservationSpaceMessage.SurroundingEntitiesEntry\x12\x15\n\rbobber_thrown\x18\x14 \x01(\x08\x12\x12\n\nexperience\x18\x15 \x01(\x05\x12\x12\n\nworld_time\x18\x16 \x01(\x03\x12\x1a\n\x12last_death_message\x18\x17 \x01(\t\x12\x0f\n\x07image_2\x18\x18 \x01(\x0c\x1a\x37\n\x15KilledStatisticsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x36\n\x14MinedStatisticsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x35\n\x13MiscStatisticsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1aS\n\x18SurroundingEntitiesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.EntitiesWithinDistance:\x02\x38\x01\x62\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "observation_space_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _OBSERVATIONSPACEMESSAGE_KILLEDSTATISTICSENTRY._options = None
    _OBSERVATIONSPACEMESSAGE_KILLEDSTATISTICSENTRY._serialized_options = b"8\001"
    _OBSERVATIONSPACEMESSAGE_MINEDSTATISTICSENTRY._options = None
    _OBSERVATIONSPACEMESSAGE_MINEDSTATISTICSENTRY._serialized_options = b"8\001"
    _OBSERVATIONSPACEMESSAGE_MISCSTATISTICSENTRY._options = None
    _OBSERVATIONSPACEMESSAGE_MISCSTATISTICSENTRY._serialized_options = b"8\001"
    _OBSERVATIONSPACEMESSAGE_SURROUNDINGENTITIESENTRY._options = None
    _OBSERVATIONSPACEMESSAGE_SURROUNDINGENTITIESENTRY._serialized_options = b"8\001"
    _ITEMSTACK._serialized_start = 27
    _ITEMSTACK._serialized_end = 138
    _BLOCKINFO._serialized_start = 140
    _BLOCKINFO._serialized_end = 209
    _ENTITYINFO._serialized_start = 212
    _ENTITYINFO._serialized_end = 347
    _HITRESULT._serialized_start = 350
    _HITRESULT._serialized_end = 503
    _HITRESULT_TYPE._serialized_start = 464
    _HITRESULT_TYPE._serialized_end = 503
    _STATUSEFFECT._serialized_start = 505
    _STATUSEFFECT._serialized_end = 581
    _SOUNDENTRY._serialized_start = 583
    _SOUNDENTRY._serialized_end = 664
    _ENTITIESWITHINDISTANCE._serialized_start = 666
    _ENTITIESWITHINDISTANCE._serialized_end = 721
    _OBSERVATIONSPACEMESSAGE._serialized_start = 724
    _OBSERVATIONSPACEMESSAGE._serialized_end = 1748
    _OBSERVATIONSPACEMESSAGE_KILLEDSTATISTICSENTRY._serialized_start = 1497
    _OBSERVATIONSPACEMESSAGE_KILLEDSTATISTICSENTRY._serialized_end = 1552
    _OBSERVATIONSPACEMESSAGE_MINEDSTATISTICSENTRY._serialized_start = 1554
    _OBSERVATIONSPACEMESSAGE_MINEDSTATISTICSENTRY._serialized_end = 1608
    _OBSERVATIONSPACEMESSAGE_MISCSTATISTICSENTRY._serialized_start = 1610
    _OBSERVATIONSPACEMESSAGE_MISCSTATISTICSENTRY._serialized_end = 1663
    _OBSERVATIONSPACEMESSAGE_SURROUNDINGENTITIESENTRY._serialized_start = 1665
    _OBSERVATIONSPACEMESSAGE_SURROUNDINGENTITIESENTRY._serialized_end = 1748
# @@protoc_insertion_point(module_scope)
