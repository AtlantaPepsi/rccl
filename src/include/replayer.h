#include <fstream>
#include "debug.h"
//#include "nccl.h"?

struct CollectiveCall {
  const char*		opName;
  uint64_t		opCount; //? vs nTasks
  const void*		sendbuff;
  void*			recvbuff;
  size_t		count;
  ncclDataType_t	datatype;
  ncclRedOp_t		op;
  int			root;
  ncclComm_t		comm;
  int			nRank;
  cudaStream_t		stream;
  int			nTasks;
  int			globalRank;
};

static char		buffer[4096];
static std::ofstream	outputFile; //nccl Comm wil take this via var, and init Replayer		
  
void			rcclReplayerInit(const char* file);
void			rcclReplayerRecord(struct CollectiveCall& coll); 
void			rcclReplayerFinish();
