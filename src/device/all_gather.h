/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
    const int bid = ncclShmem.channelId - work->channelLo;
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    size_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    size_t offset;
    size_t dataOffset;
    int nelem;
    int rankDest;

#if defined(ENABLE_NPKIT)
    int npKitCtxIdx = bid;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_ENTRY, count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, work->connIndex, work->connIndex);

#if defined(ENABLE_NPKIT)
    if (tid == 0) {
      prims.npKitCtxIdx = npKitCtxIdx;
    }
#endif

    for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
      /////////////// begin AllGather steps ///////////////
      nelem = min(chunkCount, partCount - elemOffset);
      dataOffset = partOffset + elemOffset;

      // step 0: push data to next GPU
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      if (inputBuf + dataOffset == outputBuf + offset) { // In place
        prims.directSend(dataOffset, offset, nelem);
      } else {
        prims.directCopySend(dataOffset, offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = dataOffset + rankDest * count;

        prims.directRecvCopySend(offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // Make final copy from buffer to dest.
      rankDest = ringRanks[1];
      offset = dataOffset + rankDest * count;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif
      // Final wait/copy.
      prims.directRecv(offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif



    }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_EXIT, count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t rank = ncclShmem.comm.rank;
    size_t count, gridOffset, channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    const int nThreadsBcast = work->regUsed ? (NCCL_MAX_NTHREADS - WARP_SIZE) : 4 * WARP_SIZE;
    const int nThreadsGather = work->regUsed ? WARP_SIZE : NCCL_MAX_NTHREADS - nThreadsBcast;
    const int tidEndGather = nThreadsGather;
    const int tidEndBcast = tidEndGather + nThreadsBcast;

    if (!work->regUsed) {
      if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.gather(offset, nvls->nHeads * count, nelem, count, -1, 0);
        }
      } else if (tid < tidEndBcast) {
        // Bcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndGather, nThreadsBcast, NULL, &nvls->down, work->sendbuff, NULL,
            work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.send(offset, nelem);
        }
      }
    } else {
      /* direct allgather */
      if (tid < tidEndGather) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsGather, nvls->up, nvls->up, NULL, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);

        /* used as sync */
        prims.scatter(0, 0, 0, 0, -1, 0);

        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          prims.gather(0, 0, 0, 0, -1, 0);
        }
      } else if (tid < tidEndBcast) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsBcast, &nvls->down, &nvls->down, work->sendbuff, NULL,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
        /* used as sync */
        prims.recv(0, 0);

        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          ssize_t inpOffset = gridOffset + elemOffset;
          ssize_t outOffset = inpOffset + rank * count;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(inpOffset, outOffset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool BcastSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char* inbuf = (char*)work->sendbuff;
      char* outbuf = (char*)work->recvbuff;
      ssize_t sizePerRank = work->collnet.count*sizeof(T);
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank*sizePerRank);

      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*sizePerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*sizePerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int src = 0;
      int rail;
      if (BcastSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/sizePerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*sizePerRank;
          ssize_t railOneEnd = railOneBeg + sizePerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*sizePerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ? 0 : 1;
          reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs,MinSrcs,MaxSrcs=*/0,1,1,
                     /*MultimemDsts=*/0, 0+MinDsts, 1+MaxDsts,
                     /*PreOpSrcs=*/0>
            (tid, tn, 0, nullptr, false,
             /*nSrcs=*/1, [=]__device__(int s/*==0*/) -> void* {
               return (char*)srcPtrs[src] + railAllOffset;
             },
             /*nDsts=*/outIsDst+nDsts, [=]__device__(int d) -> void* {
               return d < outIsDst ? outbuf + userOneBeg
                                   : (char*)dstPtrs[d-outIsDst] + railAllOffset;
             },
             delta);
          railAllOffset += delta;
          node += 1;
        }
        src += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (!BcastSendNotRecv && src < nRails-1);
    }
  };

  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    const int part = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t sizePerRank = work->collnet.count*sizeof(T);
    size_t chunkSize = work->collnet.chunkCount;
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = 1;
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = (isMultiRail ? 2 : 0);
    float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = work->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
        if (tid == 0) {
          int steps = (int)divUp(nNodes * sizePerRank * sizeof(T), NCCL_MAX_COLLNET_SIZE);
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, steps);
        }
        __syncwarp();
      } else {
        // Phase 1: send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, nullptr, &direct->out, work->sendbuff, nullptr,
            /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * sizePerRank; railGridOffset += nChannels * chunkSize) {
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * sizePerRank);
          ssize_t railOneBeg = ncclShmem.comm.node * sizePerRank;
          ssize_t railOneEnd = railOneBeg + sizePerRank;
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      if (work->regUsed == NCCL_COLLNET_REG_BUFFER) {
        if (tid == 0) {
          int steps = (int)divUp(nNodes * sizePerRank * sizeof(T), NCCL_MAX_COLLNET_SIZE);
          Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, steps);
        }
        __syncwarp();
      } else {
        // Phase 2: Recv network -> deposit output + send to bcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, &direct->out, direct->heads + 1, nullptr, nullptr,
            /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 0, 0);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * sizePerRank; railGridOffset += nChannels * chunkSize) {
          Scatterer</*BcastSendNotRecv=*/true> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/1>(scat);
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      // Phase 3: Recv bcast -> deposit output
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, direct->heads+1, nullptr, nullptr, nullptr,
              /*redOpArg=*/0, 2*Proto::MaxGroupWidth, 0, 0);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*BcastSendNotRecv=*/false> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.template process</*Recv=*/1, /*Send=*/0>(scat);
      }
      return;
    }
  }
};