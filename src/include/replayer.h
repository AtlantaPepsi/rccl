#include <fstream>
#include <vector>
#include "debug.h"
//#include "nccl.h"?

struct CollectiveCall {
//const char*		hostname; <- i dont know what or how to do this
  int			pid;
  int			tid;
  int			hipDev;
  ncclFunc_t		coll;
  uint64_t		opCount; //? vs nTasks
  const void*		sendbuff;
  void*			recvbuff;
  size_t		count;
  ncclDataType_t	datatype;
  ncclRedOp_t		op;
  int			root;
  ncclComm_t		comm;
  int			nRanks;
  cudaStream_t		stream;
  int			nTasks;
  int			globalRank;
};

static char		buffer[4096];
static std::ofstream	outputFile; //nccl Comm wil take this via var, and init Replayer		

static char		hostname[1024];
static int		pid = -1;
static __thread int 	tid = -1;
static int		hipDev; //get rid of?
static int		numLine = 0; //for debugging only

static std::vector<CollectiveCall> colls;
  
void			rcclReplayerInit(const char* file);
void			rcclReplayerRecord(struct CollectiveCall& coll); 
ncclResult_t		rcclReplayerWrite(ncclComm_t comm);
void			rcclReplayerFinish();
