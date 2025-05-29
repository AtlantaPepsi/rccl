#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mpi.h>
#include <fstream>
#include <unordered_set>

#include "rcclReplayer.hpp"

#include <dirent.h>
#include <stdio.h>

using namespace rccl;

static int json_format = 0; // binary by default

// move to inside class or kept as static var
static constexpr size_t rcclCallSize = sizeof(rcclApiCall);
static char line[rcclCallSize]; // size of collectivecall struct
static int lineNum = 0;
static ncclUniqueId uniqueId;

// assuming shared file system or similar
// should this be replayer or in main
static int ParseLogFormat(const char* logFormat, std::string& filename, std::string& extension)
{
  int json_format = 0;
  size_t dot;
  if ((dot = std::string(logFormat).find(".")) != std::string::npos)
  {
    filename = std::string(logFormat).substr(0, dot);
    extension = std::string(logFormat).substr(dot);
    if (extension.compare(".json") == 0)
    {
      json_format = 1;
    }
  } else {
    filename = std::string(logFormat);
  }
  return json_format;
  // ^reuse from recorder?
}

Replayer::Replayer(const std::string& logname, int json_format, int rank, int size) : myRank(rank),
                                                                                      numGlobalRanks(size)
{
  log.open(logname, json_format ? std::ifstream::in : std::ifstream::binary);
}

void Replayer::parse()
{
  while (log.read(line, rcclCallSize)) // why would get fail here when running into newline
  {
    rcclApiCall call = *((rcclApiCall*) line);

    if (call.spbase)
    {
      if (!dMemMap.contains(call.spbase))
      {
        dMemMap[call.spbase] = {.size = call.spsize};
      }
      dMemMap[call.spbase].lastLineUsed = lineNum;
    }
    if (call.rpbase)
    {
      if (!dMemMap.contains(call.rpbase))
      {
        dMemMap[call.rpbase] = {.size = call.rpsize};
      }
      dMemMap[call.rpbase].lastLineUsed = lineNum;
    }

    switch (call.type) {
    case rrGroupStart:
    case rrGroupEnd:
    case rrGroupSimulatedEnd: //TODO
    case rrCommInitRank:
    /// case rrCommInitRankConfig:   <-- these all should depend on CommInitDev
    case rrCommSplit: // <-- not covered for now dealt with in replay time
    case rrCommFinalize:
    case rrCommDestroy:
    case rrCommAbort:
    case rrCommRegister:
    case rrCommDeregister: // I think commDeregister is not affected by handle in both way?
    case rrMemFree:
    case rrRedOpCreatePreMulSum:
    case rrRedOpDestroy:
    case rrOtherCall:
    {
      break; // no op
    }
  // Communicator
    case rrGetUniqueId:
    {
      idRankMap[call.commId];
      break;
    }
    
    case rrCommInitDev:             // which should capture all comm - uniqueID relations
    {
      Ids.push_back(call.commId);
      /// commIdMap[call.commId] = call.comm; <<- do this later
      // for debugging might want a reverse map
      break;
    }
    case rrCommInitAll:
    {
      if (call.sendbuff)
      {
        log.ignore(call.root * sizeof(int));
      }
      break;
    }

  // Memory allocation
    //integrate these later
    case rrMemAlloc:
    {
      // Replayer will not free this without explicit ncclMemFree
      dMemMap[call.recvbuff] = {.size = call.count};
      break;
    }

    case rrAllToAllv:
    {
      log.ignore(4 * call.nRanks * sizeof(size_t)); // will allocate s/rdispls/count each time
    }
    default: // collectives
    {
      /*  if capturing:
       *    if first time (start.empty)
       *      init stream
       *      push this line for replayer later
       *    increment depth
       *  else
       *    use internal counter to separate diff graph launch
       */
      if (call.graphCaptured)
      {
        if (!graphLife.contains(call.graphID))
        {
          graphLife[call.graphID].starts.insert(lineNum);
          graphLife[call.graphID].stream = call.stream;
        }
        graphLife[call.graphID].depth++;
        graphLife[call.graphID].counter++;
      } else if (call.graphID != -1) {
        if (graphLife[call.graphID].counter == graphLife[call.graphID].depth)
        {
          graphLife[call.graphID].starts.insert(lineNum);
        }
        graphLife[call.graphID].counter--;
        if (graphLife[call.graphID].counter == 0)
        {
          graphLife[call.graphID].counter = graphLife[call.graphID].depth;
        }
      }
    }
    }
    lineNum++;
  }

  // exchange communicator info
  std::vector<int> comm_count(numGlobalRanks);
  comm_count[myRank] = Ids.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, comm_count.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(comm_count.size() + 1, 0);
  std::inclusive_scan(comm_count.begin(), comm_count.end(), displs.begin() + 1);
  int aggragatedCommCount = std::reduce(comm_count.begin(), comm_count.end());
  /*
   *                  rank1, comm_count[1]xID  r2, comm_count[2]  r3 ...  r4 ...
   *  AllRankCommIds [------------------------+-----------------+-------+---------+....]
   */
  std::vector<uint64_t> AllRankCommIds(aggragatedCommCount);
  MPI_Allgatherv(Ids.data(), Ids.size(), MPI_UINT64_T,
                 AllRankCommIds.data(), comm_count.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);

  int k = 0;
  for (int i = 0; i < numGlobalRanks; i++)
  {
    if (i == myRank)
    {
      k += Ids.size();
      continue;
    }
    for (int j = 0; j < comm_count[i]; j++)
    {
      if (idRankMap.contains(AllRankCommIds[k]))
      {
        idRankMap[AllRankCommIds[k]].push_back(i);
      }
      k++;
    }
  }

  lineNum = 0;
  log.clear();
  log.seekg(0, std::ios_base::beg);
}

void Replayer::replay()
{
  while (log.read(line, rcclCallSize))
  {
    rcclApiCall call = *((rcclApiCall*) line);
    hipSetDevice(call.hipDev);
    void *sbuffer = NULL, *rbuffer = NULL;

    if (call.type < rrGroupStart)
    {
      if ((call.spbase && !dMemMap.contains(call.spbase)) || (call.rpbase && !dMemMap.contains(call.rpbase)))
      {printf("ERROR %d %d\n", dMemMap.contains(call.rpbase), dMemMap.contains(call.spbase)); exit(1);}

      if (call.spbase)
      {
        if (!dMemMap[call.spbase].base)
        {
          hipMalloc(&dMemMap[call.spbase].base, dMemMap[call.spbase].size);
        }
        std::ptrdiff_t diff = (char*)call.sendbuff - (char*)call.spbase;
        sbuffer = (char*)dMemMap[call.spbase].base + diff;
      }
      if (call.rpbase)
      {
        if (!dMemMap[call.rpbase].base)
        {
          hipMalloc(&dMemMap[call.rpbase].base, dMemMap[call.rpbase].size);
        }
        std::ptrdiff_t diff = (char*)call.recvbuff - (char*)call.rpbase;
        rbuffer = (char*)dMemMap[call.rpbase].base + diff;
      }

      //stream
      if (!streams[call.stream].first) // guaranteed to be null by default?
      {
        hipStreamCreate(&streams[call.stream].first);
      }
      //graph
      /*
       *  if capturing
       *    if firstime (line in start)
       *      stream capture begin
       *    if stream differ
       *      //create dep
       *    if depth reached
       *      conclude graph
       *  else (launching)
       */
      if (call.graphCaptured)
      {
        graphLife[call.graphID].counter--;
        if (graphLife[call.graphID].starts.contains(lineNum))
        {
          hipStreamBeginCapture(streams[call.stream].first, hipStreamCaptureModeGlobal);
        } else if (graphLife[call.graphID].stream != call.stream) {
          printf("WARNING : multi-stream graph may not replay the original dependency accurately");
          hipEvent_t event;
          hipEventCreate(&event);
          graphLife[call.graphID].events.push_back(event);
          hipEventRecord(event, streams[graphLife[call.graphID].stream].first);
          hipStreamWaitEvent(streams[call.stream].first, event);
        }    
      } else if (call.graphID != -1) {
        if (graphLife[call.graphID].starts.contains(lineNum))
        {
          hipGraphLaunch(graphLife[call.graphID].graphExec, streams[call.stream].first);
        }
        goto cleanup;
      }
    }

    switch (call.type) {
    case rrGroupSimulatedEnd: //TODO
    /// case rrCommInitRankConfig:   <-- these all should depend on CommInitDev
    case rrCommSplit: // <-- not covered for now dealt with in replay time
    case rrRedOpCreatePreMulSum:
    case rrRedOpDestroy:
    case rrOtherCall:
    {
      printf("Unexpected call: %d\n", call.type);
      exit(1);
    }

    // To be integrated later
    case rrCommFinalize:
    case rrCommDestroy:
    case rrCommAbort:
    {
      ncclCommFinalize(commMap[call.comm]);
      break;
    }

    case rrGroupStart:
    {
      ncclGroupStart();
      break;
    }
    case rrGroupEnd:
    {
      ncclGroupEnd();
      break;
    }

    case rrGetUniqueId:
    {
      ncclGetUniqueId(&uniqueId);
      idMap[call.commId] = uniqueId;
      break;
    }
    case rrCommInitRank:
    {
      lastCall = rrCommInitRank;
      break;
    }
    /// case rrCommInitRankConfig:
    case rrCommInitDev:
    {
      if (lastCall == rrCommInitAll) // no other calls between ncclCommInitAll and ncclCommInitRankDev
      {                              // nor ncclCommInitRankDev not proceeded by ncclCommInitAll/Rank()
        goto cleanup;
      }
      // set device
      hipSetDevice(call.root);

      if (!idMap.contains(call.commId))
      {
        MPI_Recv(&uniqueId, sizeof(ncclUniqueId), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        for (int rank : idRankMap[call.commId])
        {
          MPI_Send(&idMap[call.commId], sizeof(ncclUniqueId), MPI_BYTE, rank, 0, MPI_COMM_WORLD);
        }
        uniqueId = idMap[call.commId]; // ?
      }
      ncclComm_t comm;
      ncclCommInitRank(&comm, call.nRanks, uniqueId, call.globalRank);
      commMap[call.comm] = comm;
      break;
    }
    case rrCommInitAll:
    {
      int ndev = call.root;
      int *devlist = NULL;
      if (call.sendbuff)
      {
        std::vector<int> devices(ndev);
        log.read((char*)devices.data(), ndev * sizeof(int));
        devlist = devices.data();
      }
      ncclComm_t comm;
      ncclCommInitAll(&comm, ndev, devlist);
      commMap[call.comm] = comm;
      break;
    }


    case rrCommRegister:
    {
      if (!dMemMap.contains(call.spbase) || !commMap.contains(call.comm)) {printf("ERROR\n"); exit(1);}
      if (!dMemMap[call.spbase].base)
      {
        hipMalloc(&dMemMap[call.spbase].base, dMemMap[call.spbase].size);
      }
      sbuffer = (char*)dMemMap[call.spbase].base + (std::ptrdiff_t)((char*)call.sendbuff - (char*)call.spbase);
      ncclCommRegister(commMap[call.comm], sbuffer, dMemMap[call.spbase].size, &handleMap[call.recvbuff]);
      break;
    }
    case rrCommDeregister:
    {
      ncclCommDeregister(commMap[call.comm], handleMap[call.recvbuff]);
      break;
    }
    case rrMemAlloc:
    {
      ncclMemAlloc(&dMemMap[call.recvbuff].base, call.count);
      break ;
    }
    case rrMemFree:
    {
      ncclMemFree(dMemMap[call.recvbuff].base);
      break;
    }

    // TODO: further simplify switch base on common parameters
    // no op or root
    case rrAllToAll:
    {
      ncclAllToAll(sbuffer, rbuffer, call.count, call.datatype, commMap[call.comm], streams[call.stream].first);
      break;
    }
    case rrAllGather:
    {
      ncclAllGather(sbuffer, rbuffer, call.count, call.datatype, commMap[call.comm], streams[call.stream].first);
      break;
    }
    // op root
    case rrReduce:
    {
      ncclReduce(sbuffer, rbuffer, call.count, call.datatype, call.op, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    // root
    case rrBroadcast:
    {
      ncclBroadcast(sbuffer, rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    case rrScatter:
    {
      ncclScatter(sbuffer, rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    case rrGather:
    {
      ncclGather(sbuffer, rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    // root -
    case rrBcast:
    {
      ncclBcast(rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    case rrSend:
    {
      ncclSend(rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    case rrRecv:
    {
      ncclRecv(rbuffer, call.count, call.datatype, call.root, commMap[call.comm], streams[call.stream].first);
      break;
    }
    // op
    case rrReduceScatter:
    {
      ncclReduceScatter(sbuffer, rbuffer, call.count, call.datatype, call.op, commMap[call.comm], streams[call.stream].first);
      break;
    }
    case rrAllReduce:
    {
      ncclAllReduce(sbuffer, rbuffer, call.count, call.datatype, call.op, commMap[call.comm], streams[call.stream].first);
      break;
    }
    // a2av
    case rrAllToAllv:
    {
      // timer pause here
      // assuming blocking for now
      int size = call.nRanks;
      std::vector<size_t> sendcounts(size), sdispls(size), recvcounts(size), rdispls(size);
      log.read((char*)sendcounts.data(), size * sizeof(size_t));
      log.read((char*)sdispls.data(), size * sizeof(size_t));
      log.read((char*)recvcounts.data(), size * sizeof(size_t));
      log.read((char*)rdispls.data(), size * sizeof(size_t));
      
      ncclAllToAllv(sbuffer, sendcounts.data(), sdispls.data(), rbuffer, recvcounts.data(), rdispls.data(),
                    call.datatype, commMap[call.comm], streams[call.stream].first);
      hipStreamSynchronize(streams[call.stream].first); // TODO: remove
      break;
    }
    }//switch
    lastCall = call.type;

    if (call.graphCaptured && graphLife[call.graphID].counter == 0)
    {
      if (graphLife[call.graphID].stream != call.stream)
      {
        hipEvent_t event;
        hipEventCreate(&event);
        graphLife[call.graphID].events.push_back(event);
        hipEventRecord(event, streams[call.stream].first);
        hipStreamWaitEvent(streams[graphLife[call.graphID].stream].first, event);
      }
      if (graphLife[call.graphID].counter == 0)
      {
        hipStreamEndCapture(graphLife[call.graphID].stream, &graphLife[call.graphID].graph);
        hipGraphInstantiate(&graphLife[call.graphID].graphExec, graphLife[call.graphID].graph, NULL, NULL, 0);
        for (hipEvent_t e : graphLife[call.graphID].events)
        {
          hipEventDestroy(e);
        }
      }
    }

cleanup:
    // Free resources if possible
    if (call.spbase && lineNum == dMemMap[call.spbase].lastLineUsed) {
      hipFree(dMemMap[call.spbase].base);
    }
    if (call.rpbase && lineNum == dMemMap[call.rpbase].lastLineUsed) {
      hipFree(dMemMap[call.rpbase].base);
    }
    if (call.stream && lineNum == streams[call.stream].second)
    {
      hipStreamSynchronize(streams[call.stream].first);
      hipStreamDestroy(streams[call.stream].first);
    }
    lineNum++; // change for a2av
  }
}

int main(int argc, char **argv)
{
  unsetenv("RCCL_REPLAY_FILE");
  MPI_Init(&argc, &argv);
  if (argc <= 1) {
    printf("Usage: %s logfile [numGpusPerMpiRank = 1]\n", argv[0]);
    exit(1);
  }

  // Parse rank information
  int mpiRank, numMpiRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numMpiRanks);

  // Parse command line arguments
  char* logFilename       = argv[1];
  int   numGpusPerMpiRank = (argc > 2 ? atoi(argv[2]) : 1);
  /// int   parseOnly         = (argc > 3 ? atoi(argv[3]) : 0);
  assert(numGpusPerMpiRank == 1);

  // Figure out starting GPU index to use based on hostname
  int nameLen, pid;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &nameLen);

  std::string output_file, output_extension;
  int json_format = ParseLogFormat(logFilename, output_file, output_extension);
  assert(json_format == 0);

  // Only root handles file-rank assignment to avoid file handle pressure
  if (mpiRank != 0)
  {
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               NULL, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
    /// MPI_Gather(&nameLen, 1, MPI_INT, NULL, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);

    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL,
                hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL,
                &pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    std::vector<char> allhosts(numMpiRanks * MPI_MAX_PROCESSOR_NAME, 0);
    std::vector<int> pids(numMpiRanks * sizeof(int), 0);

    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               allhosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    /// MPI_Gather(&nameLen, 1, MPI_INT, nameLens.data(), numMpiRanks, MPI_INT, 0, MPI_COMM_WORLD);

    // All hostnames in the recorded program
    std::unordered_set<std::string> hostnames;
    for (int i = 0; i < numMpiRanks; i++)
    {
      hostnames.insert(std::string(allhosts.data() + i * MPI_MAX_PROCESSOR_NAME)); // assuming null terminator included
    }

    // Register all hostnames and pid from recorder logs
    std::unordered_map<std::string, std::vector<int>> logHosts;
    int file_pid, a = 0/*counter*/;
    DIR *d;
    struct dirent *dir;
    if (d = opendir(".")) {
      while ((dir = readdir(d)) != NULL) {
        if (sscanf(dir->d_name, (output_file + ".%d.%256[^.]" + output_extension).c_str(), &file_pid, hostname) == 2)
        {
          logHosts[std::string(hostname)].push_back(file_pid);
          a++;
        }
      }
      closedir(d);
    }
    // Double check number of nodes and number of processes match for recorder and replayer
    assert(logHosts.size() == hostnames.size());
    assert(a == numMpiRanks);
    // Assign mapping of replayer hostname to recorder hostname
    std::unordered_map<std::string, std::string> hostAssignment;
    auto it = logHosts.begin();
    for (const auto &host : hostnames)
    {
      hostAssignment[host] = (*it).first;
      it++;
    }
    for (int i = 0; i < numMpiRanks; i++)
    {
      std::string host(allhosts.data() + i * MPI_MAX_PROCESSOR_NAME);
      strcpy(allhosts.data() + i * MPI_MAX_PROCESSOR_NAME, hostAssignment[host].c_str());
      pids[i] = logHosts[hostAssignment[host]].back();
      logHosts[hostAssignment[host]].pop_back();
    }

    // Distribute the target log for each rank (pid and hostname)
    MPI_Scatter(allhosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(pids.data(), 1, MPI_INT,
                &pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  // Initialize Replayer
  std::string logfile = output_file + "." + std::to_string(pid) + "." +
                        std::string(hostname) + output_extension; /// perhaps another func for assemble logname
  std::cout << mpiRank << " : " << logfile<<std::endl;
  Replayer replayer(logfile, json_format, mpiRank, numMpiRanks);

  if (mpiRank == 0)
    printf("RCCL Replayer version 0: %d ranks x %d gpu/Rank\n", numMpiRanks, numGpusPerMpiRank);
  printf("Rank %d [%s]\n", mpiRank, hostname);

  replayer.parse();
  printf("rank %d parsing completed, starting replay\n", mpiRank);
  replayer.replay();
  MPI_Finalize();
  return 0;
}
