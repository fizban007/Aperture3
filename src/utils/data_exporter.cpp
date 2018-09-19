#include "utils/data_exporter.h"
#include <cstring>
#include <fstream>
#include <time.h>
#include <utility>

using namespace Aperture;

DataExporter::DataExporter() { SetDefault(); }

DataExporter::DataExporter(const std::string& dir,
                           const std::string& prefix, bool compress,
                           int rank, int size) {
  SetDefault();
  numRanks = size;
  myRank = rank;

  boost::filesystem::path rootPath(dir.c_str());
  boost::system::error_code returnedError;

  boost::filesystem::create_directories(rootPath, returnedError);

  if (returnedError)
    std::cerr << "Error reading directory!" << std::endl;

  outputDirectory = dir;
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');
  char myTime[100] = {};
  char subDir[100] = {};
  time_t rawtime;
  struct tm* timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(myTime, 100, "%Y%m%d-%H%M", timeinfo);
  snprintf(subDir, sizeof(subDir), "Data%s/", myTime);

  outputDirectory += subDir;
  subName = "rank" + std::to_string(myRank) + "/";
  subDirectory = outputDirectory + subName;

  boost::filesystem::path subPath(subDirectory);

  boost::filesystem::create_directories(subPath, returnedError);

  if (returnedError)
    std::cerr << "Error creating sub-directory!" << std::endl;

  filePrefix = prefix;
  useCompression = compress;
}

DataExporter::~DataExporter() {
  if (lowOffset != nullptr) delete[] lowOffset;
  if (hiOffset != nullptr) delete[] hiOffset;
}

DataExporter&
DataExporter::operator=(DataExporter&& other) {
  outputDirectory = other.outputDirectory;
  subDirectory = other.subDirectory;
  subName = other.subName;
  filePrefix = other.filePrefix;
  myRank = other.myRank;
  numRanks = other.numRanks;

  quadMeshes = std::move(other.quadMeshes);
  quadVarsF = std::move(other.quadVarsF);
  quadVarsD = std::move(other.quadVarsD);
  dbVars = std::move(other.dbVars);
  dbArrays = std::move(other.dbArrays);

  numFields = other.numFields;
  useCompression = other.useCompression;

  if (lowOffset != nullptr) delete[] lowOffset;
  if (hiOffset != nullptr) delete[] hiOffset;
  lowOffset = other.lowOffset;
  hiOffset = other.hiOffset;
  other.lowOffset = nullptr;
  other.hiOffset = nullptr;

  return *this;
}

void
DataExporter::SetDefault() {
  lowOffset = new int[3];
  hiOffset = new int[3];
  outputDirectory = "data/";
  filePrefix = "output";
  useCompression = true;
  numFields = 0;
}

void
DataExporter::AddField(const char* name, const float* data,
                       const char* mesh) {
  silo_quadVariable<float> tempVar;

  tempVar.meshName = mesh;
  tempVar.quadName = name;
  tempVar.data = data;

  for (int i = 0; i < (int)quadMeshes.size(); i++) {
    if (quadMeshes[i].meshName == std::string(mesh))
      tempVar.mesh = &quadMeshes[i];
  }

  quadVarsF.push_back(tempVar);
}

void
DataExporter::AddField(const char* name, const double* data,
                       const char* mesh) {
  silo_quadVariable<double> tempVar;

  tempVar.meshName = mesh;
  tempVar.quadName = name;
  tempVar.data = data;

  for (int i = 0; i < (int)quadMeshes.size(); i++) {
    if (quadMeshes[i].meshName == std::string(mesh))
      tempVar.mesh = &quadMeshes[i];
  }

  quadVarsD.push_back(tempVar);
}

void
DataExporter::AddArray(const char* name, float* data, int* dims,
                       int ndims) {
  silo_dbArray tempArray;

  tempArray.varName = name;
  tempArray.data = data;
  tempArray.ndims = ndims;
  // tempArray.dims = dims;
  for (int i = 0; i < ndims; i++) tempArray.dims.push_back(dims[i]);

  dbArrays.push_back(tempArray);
}

void
DataExporter::AddVariable(const char* name, float* data) {
  silo_dbVariable tempVar;

  tempVar.varName = name;
  tempVar.data = data;

  dbVars.push_back(tempVar);
}

void
DataExporter::WriteOutput(int timeStep, float time, const Index& pos,
                          bool displayGuard) {
  DBfile* dbfile = nullptr;

  char filenameMaster[50];
  char filename[50];
  int dim_dbVar[] = {1};

  boost::filesystem::path rootPath(outputDirectory.c_str());
  boost::system::error_code returnedError;

  if (!boost::filesystem::exists(rootPath)) {
    boost::filesystem::create_directories(rootPath, returnedError);
    if (returnedError)
      std::cerr << "Error creating directory!" << std::endl;
  }

  // This is the main file if there are multiple ranks
  sprintf(filenameMaster, "%s%s%05d.silo", outputDirectory.c_str(),
          filePrefix.c_str(), timeStep);
  sprintf(filename, "%s%s%05d.d", subDirectory.c_str(),
          filePrefix.c_str(), timeStep);

  DBoptlist* optlist = DBMakeOptlist(8);
  DBAddOption(optlist, DBOPT_TIME, &time);
  DBAddOption(optlist, DBOPT_CYCLE, &timeStep);
  if (displayGuard) {
    DBAddOption(optlist, DBOPT_LO_OFFSET, (void*)lowOffset);
    DBAddOption(optlist, DBOPT_HI_OFFSET, (void*)hiOffset);
  }
  // int pos[3] = { domain -> pos().x, domain -> pos().y, domain ->
  // pos().z };
  int pos_c[3] = {pos.x, pos.y, pos.z};
  DBAddOption(optlist, DBOPT_BASEINDEX, pos_c);

  DBSetFriendlyHDF5Names(1);
  if (useCompression) DBSetCompression("METHOD=GZIP");

  // char* names[] = {"Efield"};
  // char* defs[] = {"{Er, Ez}"};
  // int type[] = {DB_VARTYPE_VECTOR};

  dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL,
                    "Output of PIC simulation", DB_HDF5);
  // for (quadMesh mesh : quadMeshes) {
  for (int i = 0; i < (int)quadMeshes.size(); i++) {
    auto& mesh = quadMeshes[i];
    if (mesh.linear)
      DBPutQuadmesh(dbfile, mesh.meshName.c_str(), NULL, mesh.gridArray,
                    mesh.sizeOfDims.data(), mesh.numDim, DB_FLOAT,
                    DB_COLLINEAR, optlist);
    else
      DBPutQuadmesh(dbfile, mesh.meshName.c_str(), NULL, mesh.gridArray,
                    mesh.sizeOfDims.data(), mesh.numDim, DB_FLOAT,
                    DB_NONCOLLINEAR, optlist);
  }
  // DBPutDefvars(dbfile, "defvars", 1, names, type, defs, NULL);

  // for (quadVariable var : quadVars)
  for (int i = 0; i < (int)quadVarsF.size(); i++) {
    auto& var = quadVarsF[i];
    DBPutQuadvar1(dbfile, var.quadName.c_str(), var.meshName.c_str(),
                  var.data, var.mesh->sizeOfDims.data(),
                  var.mesh->numDim, NULL, 0, DB_FLOAT, DB_NODECENT,
                  NULL);
  }

  for (int i = 0; i < (int)quadVarsD.size(); i++) {
    auto& var = quadVarsD[i];
    DBPutQuadvar1(dbfile, var.quadName.c_str(), var.meshName.c_str(),
                  var.data, var.mesh->sizeOfDims.data(),
                  var.mesh->numDim, NULL, 0, DB_DOUBLE, DB_NODECENT,
                  NULL);
  }

  // for (dbVariable var : dbVars)

  DBFreeOptlist(optlist);
  DBClose(dbfile);
  // Only do master file if in processor 0
  if (myRank == 0) {
    DBfile* dbfileMaster = nullptr;
    dbfileMaster = DBCreate(filenameMaster, DB_CLOBBER, DB_LOCAL,
                            "Master file of PIC simulation", DB_HDF5);

    for (int i = 0; i < (int)dbVars.size(); i++) {
      auto& var = dbVars[i];
      DBWrite(dbfileMaster, var.varName.c_str(), var.data, dim_dbVar, 1,
              DB_FLOAT);
    }

    for (int i = 0; i < (int)dbArrays.size(); i++) {
      auto& var = dbArrays[i];
      DBWrite(dbfileMaster, var.varName.c_str(), var.data,
              var.dims.data(), var.ndims, DB_FLOAT);
    }

    for (unsigned int i = 0; i < quadMeshes.size(); i++) {
      std::vector<char*> meshnames;
      std::vector<int> meshtypes;

      for (int j = 0; j < numRanks; j++) {
        char* name = new char[100];
        sprintf(name, "%s%d/%s%05d.d:%s", "rank", j, filePrefix.c_str(),
                timeStep, quadMeshes[i].meshName.c_str());
        meshnames.push_back(name);
        meshtypes.push_back(
            (quadMeshes[i].linear ? DB_QUAD_RECT : DB_QUAD_CURV));
        // std::cout << name << std::endl;
      }

      DBPutMultimesh(dbfileMaster, quadMeshes[i].meshName.c_str(),
                     numRanks, meshnames.data(), meshtypes.data(),
                     NULL);

      for (int j = 0; j < numRanks; j++) {
        delete[] meshnames[j];
      }
    }

    for (unsigned int i = 0; i < quadVarsF.size(); i++) {
      std::vector<char*> varnames;
      std::vector<int> vartypes;

      for (int j = 0; j < numRanks; j++) {
        char* name = new char[100];
        sprintf(name, "%s%d/%s%05d.d:%s", "rank", j, filePrefix.c_str(),
                timeStep, quadVarsF[i].quadName.c_str());
        varnames.push_back(name);
        vartypes.push_back(DB_QUADVAR);
      }

      DBPutMultivar(dbfileMaster, quadVarsF[i].quadName.c_str(),
                    numRanks, varnames.data(), vartypes.data(), NULL);

      for (int j = 0; j < numRanks; j++) {
        delete[] varnames[j];
      }
    }

    for (unsigned int i = 0; i < quadVarsD.size(); i++) {
      std::vector<char*> varnames;
      std::vector<int> vartypes;

      for (int j = 0; j < numRanks; j++) {
        char* name = new char[100];
        sprintf(name, "%s%d/%s%05d.d:%s", "rank", j, filePrefix.c_str(),
                timeStep, quadVarsD[i].quadName.c_str());
        varnames.push_back(name);
        vartypes.push_back(DB_QUADVAR);
      }

      DBPutMultivar(dbfileMaster, quadVarsD[i].quadName.c_str(),
                    numRanks, varnames.data(), vartypes.data(), NULL);

      for (int j = 0; j < numRanks; j++) {
        delete[] varnames[j];
      }
    }
    DBClose(dbfileMaster);
  }
}

void
DataExporter::CopyConfig(const std::string& file) {
  if (myRank != 0) return;
  boost::filesystem::path rootPath(outputDirectory.c_str());
  boost::system::error_code returnedError;

  if (!boost::filesystem::exists(rootPath)) {
    boost::filesystem::create_directories(rootPath, returnedError);
    if (returnedError)
      std::cerr << "Error creating directory!" << std::endl;
  }

  boost::filesystem::path file_path(file.c_str());
  boost::filesystem::path target_path(
      (outputDirectory + file_path.filename().string()).c_str());

  boost::filesystem::copy_file(file_path, target_path, returnedError);
  if (returnedError)
    std::cerr << "Error copying config file!" << std::endl;
}

void
DataExporter::CopyMain() {
  if (myRank != 0) return;
  boost::filesystem::path rootPath(outputDirectory.c_str());
  boost::system::error_code returnedError;

  if (!boost::filesystem::exists(rootPath)) {
    boost::filesystem::create_directories(rootPath, returnedError);
    if (returnedError)
      std::cerr << "Error creating directory!" << std::endl;
  }

  std::string mainfile = "../src/main.cpp";

  boost::filesystem::path file_path(mainfile.c_str());
  boost::filesystem::path target_path(
      (outputDirectory + "main.cpp").c_str());

  boost::filesystem::copy_file(file_path, target_path, returnedError);
  if (returnedError)
    std::cerr << "Error copying main.cpp!" << std::endl;
}
