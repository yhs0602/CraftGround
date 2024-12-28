#ifndef __IPC_APPLE_H__
#define __IPC_APPLE_H__

#include "dlpack.h"
DLManagedTensor *mtl_tensor_from_mach_port(int machPort, int width, int height);

#endif // __IPC_APPLE_H__