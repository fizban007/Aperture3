#include "utils/silo_file.h"
#include "core/grid.h"
#include "fmt/format.h"
#include "metrics.h"
#include <boost/filesystem.hpp>
#include <fstream>

using namespace Aperture;

silo_file::silo_file() {
  for (int i = 0; i < 3; i++) {
    m_dims[i] = 1;
    m_domainDecomp[i] = 1;
    m_domainSize[i] = 1;
    m_guard[i] = 0;
  }
  m_numRanks = 1;
  // m_grid_conf.resize(3);
}

silo_file::~silo_file() {
  if (m_open) {
    close();
  }
}

void
silo_file::open_file(const std::string& filename) {
  // Don't double open
  if (m_open) close();
  m_filename = filename;
  // boost::filesystem::path file_path(filename);

  fmt::print("Trying to open {}\n", filename);
  m_dbfile = DBOpen(filename.c_str(), DB_HDF5, DB_READ);

  if (!m_dbfile) {
    fmt::print(stderr, "File open failed!\n");
    m_open = false;
    return;
  }

  std::ifstream fs;
  boost::filesystem::path parent_dir(filename);
  parent_dir.remove_filename();
  m_parent_dir = parent_dir.string();
  boost::filesystem::path grid_file_path = parent_dir / "grid_config";
  // fmt::print("Config file path is {}\n", grid_file_path.string());
  fs.open(grid_file_path.string());
  if (fs) {
    for (int i = 0; i < 3; i++) {
      std::getline(fs, m_grid_conf[i]);
    }
  } else {
    fmt::print(stderr,
               "Open grid config file error, skipping the grid config");
  }
  fs.close();

  // Read the table of contents
  m_dbToc = DBGetToc(m_dbfile);
  if (!m_dbToc) {
    fmt::print(stderr, "Error reading the toc!\n");
    m_open = false;
    DBClose(m_dbfile);
    m_dbfile = nullptr;
    return;
  }

  m_isMultimesh = false;
  // Check if directory rank0 or directory group0 exists. If it exists
  // then we use multimesh, otherwise we use normal method
  boost::filesystem::path rankPath = parent_dir / "rank0";
  boost::filesystem::path groupPath = parent_dir / "group0";
  if (boost::filesystem::exists(rankPath) ||
      boost::filesystem::exists(groupPath)) {
    m_isMultimesh = true;
  }

  if (m_isMultimesh) {
    // TODO: Finish this part!!!
    m_multiMesh = DBGetMultimesh(m_dbfile, m_dbToc->multimesh_names[0]);
    // number of ranks in the file
    m_numRanks = m_multiMesh->nblocks;
    m_pos.resize(m_numRanks);

    // Read in the quad meshes and determine the geometry of the ranks
    // TODO: Tune this to be compatible with groups
    for (int i = 0; i < m_numRanks; i++) {
      std::string subName(m_multiMesh->meshnames[i]);
      std::size_t found = subName.find_last_of(":");
      std::string subFile =
          m_parent_dir + "/" + subName.substr(0, found);
      std::string meshName = subName.substr(found + 1);

      DBfile* dbSubFile = DBOpen(subFile.c_str(), DB_HDF5, DB_READ);
      m_dbQuadmeshes.push_back(
          DBGetQuadmesh(dbSubFile, meshName.c_str()));
      // std::cout << "(" << dbQuadMeshes.back() -> base_index[0] << ",
      // "
      //           << dbQuadMeshes.back() -> base_index[1] << ", "
      //           << dbQuadMeshes.back() -> base_index[2] << ")" <<
      //           std::endl;

      // domain size is the number of domains in each direction
      m_domainDecomp[0] = std::max(
          m_dbQuadmeshes.back()->base_index[0] + 1, m_domainDecomp[0]);
      m_domainDecomp[1] = std::max(
          m_dbQuadmeshes.back()->base_index[1] + 1, m_domainDecomp[1]);
      m_domainDecomp[2] = std::max(
          m_dbQuadmeshes.back()->base_index[2] + 1, m_domainDecomp[2]);
      m_pos[i].x = m_dbQuadmeshes[i]->base_index[0];
      m_pos[i].y = m_dbQuadmeshes[i]->base_index[1];
      m_pos[i].z = m_dbQuadmeshes[i]->base_index[2];
      DBClose(dbSubFile);
    }
    for (int i = 0; i < 3; i++) {
      // The ghost zone in the output file is guard cell - 1
      m_guard[i] = m_dbQuadmeshes.back()->min_index[i] + 1;
      if (m_dbQuadmeshes.back()->size_index[i] > 0) {
        m_domainSize[i] =
            m_dbQuadmeshes.back()->size_index[i] - 2 * m_guard[i];
        m_dims[i] =
            m_domainSize[i] * m_domainDecomp[i] + 2 * m_guard[i];
      } else {
        m_domainSize[i] = 1;
        m_dims[i] = 1;
      }
    }
  } else {
    m_numRanks = 1;
    m_pos.resize(1);
    m_dbQuadmeshes.push_back(DBGetQuadmesh(m_dbfile, "quadmesh"));
    for (int i = 0; i < 3; i++) {
      if (m_dbQuadmeshes[0]->size_index[i] > 0) {
        m_dims[i] = m_dbQuadmeshes.back()->size_index[i];
        m_guard[i] = m_dbQuadmeshes.back()->min_index[i] + 1;
      } else {
        m_dims[i] = 1;
        m_guard[i] = 0;
      }
      m_domainSize[i] = m_dims[i] - 2 * m_guard[i];
      m_domainDecomp[i] = 1;
      m_pos[0] = Index(0, 0, 0);
    }
  }

  m_open = true;
}

void
silo_file::close() {
  // Don't close if there is no open file
  if (!m_open) return;

  if (m_dbfile) DBClose(m_dbfile);
  m_dbfile = nullptr;
  m_dbToc = nullptr;
  for (unsigned int i = 0; i < m_dbQuadmeshes.size(); i++) {
    if (m_dbQuadmeshes[i]) DBFreeQuadmesh(m_dbQuadmeshes[i]);
    m_dbQuadmeshes[i] = nullptr;
  }
  if (m_multiMesh) DBFreeMultimesh(m_multiMesh);
  m_multiMesh = nullptr;

  m_open = false;
}

void
silo_file::show_content() {
  if (!m_dbfile) {
    fmt::print(stderr, "File not open!\n");
    return;
  }

  fmt::print("Normal variables:\n");
  for (int i = 0; i < m_dbToc->nvar; i++) {
    fmt::print("{}\n", m_dbToc->var_names[i]);
  }
  fmt::print("Multi-variables:\n");
  for (int i = 0; i < m_dbToc->nmultivar; i++) {
    fmt::print("{}\n", m_dbToc->multivar_names[i]);
  }
}

void
silo_file::get_dims(int dims[3]) const {
  for (int i = 0; i < 3; i++) dims[i] = m_dims[i];
}

void
silo_file::get_domain_size(int sizes[3]) const {
  for (int i = 0; i < 3; i++) {
    if (m_domainSize[i] > 1)
      sizes[i] = m_domainSize[i] + 2 * m_guard[i];
    else
      sizes[i] = m_domainSize[i];
  }
}

cu_multi_array<float>
silo_file::get_quad_var(const std::string& varname) const {
  cu_multi_array<float> result;
  if (m_numRanks != 1 || m_isMultimesh) {
    fmt::print(stderr,
               "We have multimesh instead of quadmesh, use "
               "get_multi_var instead!\n");
    return result;
  }

  DBquadvar* quadVar;
  // std::cout << rank << " " << varname << std::endl;
  quadVar = DBGetQuadvar(m_dbfile, varname.c_str());
  if (quadVar == NULL)
    std::cout << "error in get_quad_var unranked" << std::endl;

  // int dims[3] = {m_domainSize[0] + 2 * m_guard[0], m_domainSize[1] +
  // 2 * m_guard[1],
  //                m_domainSize[2] + 2 * m_guard[2]};
  // if (_domainSize[2] == 1) dims[2] = 1;
  // if (_domainSize[1] == 1) dims[1] = 1;
  // if (_domainSize[0] == 1) dims[0] = 1;
  // array->values = new float[dims[0] * dims[1] * dims[2]];
  result.resize(Extent(m_dims[0], m_dims[1], m_dims[2]));
  for (int k = 0; k < m_dims[2]; k++) {
    for (int j = 0; j < m_dims[1]; j++) {
      for (int i = 0; i < m_dims[0]; i++) {
        int idx = i + j * m_dims[0] + k * m_dims[0] * m_dims[1];
        // float** ptr = (float**)quadVar->vals;
        // array->values[idx] = ptr[0][idx];
        if (quadVar->datatype == DB_DOUBLE) {
          result[idx] = (float)(((double**)quadVar->vals)[0][idx]);
        } else if (quadVar->datatype == DB_FLOAT) {
          result[idx] = ((float**)quadVar->vals)[0][idx];
        }
      }
    }
  }
  DBFreeQuadvar(quadVar);
  return result;
}

cu_multi_array<float>
silo_file::get_quad_var(const std::string& varname, int rank) const {
  cu_multi_array<float> result;
  if (rank < 0 || rank > m_numRanks) {
    fmt::print(stderr, "Unavailable rank!");
    return result;
  }
  DBquadvar* quadVar;

  std::string subName(m_multiMesh->meshnames[rank]);
  std::size_t found = subName.find_last_of(":");
  std::string subFile = m_parent_dir + "/" + subName.substr(0, found);

  DBfile* dbSubFile = DBOpen(subFile.c_str(), DB_HDF5, DB_READ);
  // _dbQuadMeshes.push_back(DBGetQuadmesh(dbSubFile,
  // meshName.c_str()));
  quadVar = DBGetQuadvar(dbSubFile, varname.c_str());
  if (quadVar == NULL)
    std::cout << "error in get_quad_var ranked" << std::endl;

  DBClose(dbSubFile);

  int dims[3] = {m_domainSize[0] + 2 * m_guard[0],
                 m_domainSize[1] + 2 * m_guard[1],
                 m_domainSize[2] + 2 * m_guard[2]};
  if (m_domainSize[2] == 1) dims[2] = 1;
  if (m_domainSize[1] == 1) dims[1] = 1;
  if (m_domainSize[0] == 1) dims[0] = 1;
  result.resize(dims[0], dims[1], dims[2]);
  for (int k = 0; k < dims[2]; k++) {
    for (int j = 0; j < dims[1]; j++) {
      for (int i = 0; i < dims[0]; i++) {
        int idx = i + j * dims[0] + k * dims[0] * dims[1];
        float** ptr = (float**)quadVar->vals;
        result[idx] = ptr[0][idx];
      }
    }
  }
  DBFreeQuadvar(quadVar);
  return result;
}

cu_multi_array<float>
silo_file::get_multi_var(const std::string& varname) const {
  cu_multi_array<float> result;
  if (!m_isMultimesh) {
    return get_quad_var(varname);
  }

  std::vector<DBquadvar*> dbQuadVars;
  DBmultivar* dbMultiVar;
  dbMultiVar = DBGetMultivar(m_dbfile, varname.c_str());

  // Read the multivars
  // TODO: tune this for groups
  for (int n = 0; n < m_numRanks; n++) {
    std::string subName(dbMultiVar->varnames[n]);
    std::size_t found = subName.find_last_of(":");
    std::string subFile = m_parent_dir + "/" + subName.substr(0, found);
    std::string varName = subName.substr(found + 1);

    std::cout << "Opening subfile " << subFile << std::endl;
    DBfile* dbSubFile = DBOpen(subFile.c_str(), DB_HDF5, DB_READ);
    if (!dbSubFile) fmt::print(stderr, "Error opening subfile!!!\n");
    dbQuadVars.push_back(DBGetQuadvar(dbSubFile, varName.c_str()));
    DBClose(dbSubFile);
  }

  // Allocate a big array to hold the result
  result.resize(Extent(m_dims[0], m_dims[1], m_dims[2]));

  int sizes[3] = {m_domainSize[0], m_domainSize[1], m_domainSize[2]};
  if (sizes[0] > 1) sizes[0] += m_guard[0] * 2;
  if (sizes[1] > 1) sizes[1] += m_guard[1] * 2;
  if (sizes[2] > 1) sizes[2] += m_guard[2] * 2;

  for (int n = 0; n < m_numRanks; n++) {
    // std::cout << "(" << _meshPos[n].x << ", " << _meshPos[n].y << ",
    // " << _meshPos[n].z << ")" << std::endl;
    for (int k = 0; k < sizes[2]; k++) {
      for (int j = 0; j < sizes[1]; j++) {
        for (int i = 0; i < sizes[0]; i++) {
          int idx_sub = i + j * sizes[0] + k * sizes[0] * sizes[1];
          int idx_arr = i + m_pos[n].x * m_domainSize[0] +
                        (j + m_pos[n].y * m_domainSize[1]) * m_dims[0] +
                        (k + m_pos[n].z * m_domainSize[2]) * m_dims[0] *
                            m_dims[1];
          if (dbQuadVars[n]->datatype == DB_DOUBLE) {
            result[idx_arr] =
                (float)(((double**)dbQuadVars[n]->vals)[0][idx_sub]);
          } else if (dbQuadVars[n]->datatype == DB_FLOAT) {
            result[idx_arr] =
                ((float**)dbQuadVars[n]->vals)[0][idx_sub];
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < dbQuadVars.size(); i++)
    DBFreeQuadvar(dbQuadVars[i]);
  DBFreeMultivar(dbMultiVar);

  return result;
}

cu_multi_array<float>
silo_file::get_multi_mesh(int comp) const {
  cu_multi_array<float> result(m_dims[0], m_dims[1], m_dims[2]);

  int sizes[3] = {m_domainSize[0], m_domainSize[1], m_domainSize[2]};
  if (sizes[0] > 1) sizes[0] += m_guard[0] * 2;
  if (sizes[1] > 1) sizes[1] += m_guard[1] * 2;
  if (sizes[2] > 1) sizes[2] += m_guard[2] * 2;

  for (int n = 0; n < m_numRanks; n++) {
    for (int k = 0; k < sizes[2]; k++) {
      for (int j = 0; j < sizes[1]; j++) {
        for (int i = 0; i < sizes[0]; i++) {
          int idx_sub = i + j * sizes[0] + k * sizes[0] * sizes[1];
          int idx_arr = i + m_pos[n].x * m_domainSize[0] +
                        (j + m_pos[n].y * m_domainSize[1]) * m_dims[0] +
                        (k + m_pos[n].z * m_domainSize[2]) * m_dims[0] *
                            m_dims[1];
          if (m_dbQuadmeshes[n]->datatype == DB_FLOAT) {
            result[idx_arr] =
                ((float**)m_dbQuadmeshes[n]->coords)[comp][idx_sub];
          } else if (m_dbQuadmeshes[n]->datatype == DB_DOUBLE) {
            result[idx_arr] =
                ((double**)m_dbQuadmeshes[n]->coords)[comp][idx_sub];
          }
        }
      }
    }
  }
  return result;
}

cu_multi_array<float>
silo_file::get_coord_array(int dir) const {
  cu_multi_array<float> result(m_dims[0], m_dims[1], m_dims[2]);
  Grid g(m_grid_conf);
  // select_metric(g.setup_metric, parse_metric(metric), g);
  auto mesh = g.mesh();
  for (int k = 0; k < m_dims[2]; k++) {
    auto pos_k = mesh.pos(2, k, false);
    // int idx_k = k * m_dims[1] * m_dims[0];
    for (int j = 0; j < m_dims[1]; j++) {
      auto pos_j = mesh.pos(1, j, false);
      // int idx_j = j * m_dims[0];
      for (int i = 0; i < m_dims[0]; i++) {
        auto pos_i = mesh.pos(0, i, false);
        if (dir == 0)
          result(i, j, k) = pos_i;
        else if (dir == 1)
          result(i, j, k) = pos_j;
        else if (dir == 2)
          result(i, j, k) = pos_k;
      }
    }
  }
  return result;
}

cu_multi_array<float>
silo_file::get_raw_array(const std::string& varname) const {
  cu_multi_array<float> result;

  // First query whether the variable is present in the database
  if (DBInqVarExists(m_dbfile, varname.c_str()) == 0) {
    fmt::print(stderr, "Requested variable does not exist!\n");
    return result;
  }
  // Get the pointer to the data and allocate the memory for it
  void* ptr = DBGetVar(m_dbfile, varname.c_str());
  // Get the type and length of the data array
  auto type = DBGetVarType(m_dbfile, varname.c_str());
  int l = DBGetVarLength(m_dbfile, varname.c_str());
  result.resize(l);

  for (int i = 0; i < l; i++) {
    if (type == DB_DOUBLE)
      result[i] = ((double*)ptr)[i];
    else if (type == DB_FLOAT)
      result[i] = ((float*)ptr)[i];
    else if (type == DB_INT)
      result[i] = ((int*)ptr)[i];
  }

  return result;
}

bool
silo_file::find_var(const std::string& varname) const {
  if (DBInqVarExists(m_dbfile, varname.c_str()) != 0)
    return true;
  else
    return false;
}
