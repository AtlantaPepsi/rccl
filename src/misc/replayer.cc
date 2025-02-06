#include "group.h"

void rcclReplayerInit(const char* file)
{
  if (file)
    outputFile.open(file, std::ofstream::binary);
}

void rcclReplayerRecord(struct CollectiveCall& coll)
{
  if(outputFile.is_open())
  {
    const char *fmt = "%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d \
                       op %d root %d comm %p [nranks=%d] stream %p task %d globalrank %d\n";
    snprintf(buffer, 4096, fmt,
             coll.opName, coll.opCount, coll.sendbuff, coll.recvbuff, coll.count, coll.datatype,
             coll.op, 0, coll.comm, coll.comm->nRanks, coll.stream, coll.nTasks, coll.globalRank);
    outputFile.write(buffer, 4096);
  }
}

void rcclReplayerFinish()
{
  if(outputFile.is_open())
    outputFile.close();
}
