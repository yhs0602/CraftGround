#pragma once
#include <mach/mach.h>

// The kernel translates the descriptor into the receiver's port namespace.
struct IOSurfaceMessage {
    mach_msg_header_t header;
    mach_msg_body_t body;
    mach_msg_port_descriptor_t surface;
};
constexpr mach_msg_id_t IOSurfaceMessageId = 0x43475346;
