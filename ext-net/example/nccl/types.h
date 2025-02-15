/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_TYPES_H_
#define NCCL_TYPES_H_

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclBfloat16   = 9,
               ncclFp8E4M3    = 10,
               ncclFp8E5M2    = 11,
} ncclDataType_t;

#endif
