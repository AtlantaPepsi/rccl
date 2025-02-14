#include "bootstrap.h"
#include "group.h"
#include "utils.h"
#include <algorithm>
#include <sys/syscall.h> //this versus plain gettid()?

void rcclReplayerInit(const char* file)
{
  // each comm should open and close, assuming no overlapped comm
  if (!file)
    return;

  outputFile.open(file, std::ofstream::binary);
  getHostName(hostname, 1024, '.');
  pid = getpid();
}

void rcclReplayerRecord(struct CollectiveCall& coll)
{
  if (!outputFile.is_open())
    return;

  if (tid == -1)
    tid = syscall(SYS_gettid);
  hipGetDevice(&hipDev);

  coll.pid = pid;
  coll.tid = tid;
  coll.hipDev = hipDev;
  colls.push_back(coll); //move needed?
}

ncclResult_t rcclReplayerWrite(ncclComm_t comm)
{
  ncclResult_t ret = ncclSuccess;
  if (!outputFile.is_open())
    return ret;

  int rank = colls[0].globalRank;
  int nRanks = colls[0].nRanks;
  int size = colls.size();
  colls.resize(nRanks * size);
  //i wonder how expensive this would be vs default sized buffer
  std::swap_ranges(colls.begin(), colls.begin() + size, colls.begin() + rank * size);

  //this should be blocking based on socketProgress
  printf("%d : start; %d %d\n",rank,(&colls[0])->nRanks, (&colls[0]+rank*size)->nRanks);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, &colls[0], size * sizeof(CollectiveCall)), ret, fail);
  printf("%d : end; %d %d\n",rank,(&colls[0])->nRanks, (&colls[0]+rank*size)->nRanks);


  if (rank == 0)
  {
    for (int i = 0; i < colls.size(); i++)
    {
      //concat string
      CollectiveCall& coll = colls[i];

      //host/process is wrong, also write header about how many things for each rank and global info
     const char *fmt = "%s:%d:%d [%d]: \
                         op #%d: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d \
                         op %d root %d comm %p [nranks=%d] stream %p task %d globalrank %d\n";
      snprintf(buffer, 4096, fmt, hostname, coll.pid, coll.tid, coll.hipDev,
               coll.coll/*naming*/, coll.opCount, coll.sendbuff, coll.recvbuff, coll.count, coll.datatype,
               coll.op, 0, coll.comm, coll.nRanks, coll.stream, coll.nTasks, coll.globalRank);
      outputFile.write(buffer, 4096);
      numLine++;
    }
  }

exit:
  return ret;
fail:
  printf("oh no whats going on\n");
  goto exit;
}

void rcclReplayerFinish()
{
  if(outputFile.is_open())
  {
    outputFile.close();
    colls.clear();
  }
}
