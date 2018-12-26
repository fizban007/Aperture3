#ifndef _SILO_FILE_H_
#define _SILO_FILE_H_

#include "data/fields_dev.h"
#include "data/multi_array_dev.h"
#include "data/vec3.h"
#include <array>
#include <silo.h>
#include <string>
#include <vector>

namespace Aperture {

class silo_file {
 public:
  silo_file();
  ~silo_file();

  void open_file(const std::string& filename);
  void close();

  void show_content();

  void get_dims(int dims[3]) const;
  void get_domain_size(int sizes[3]) const;

  multi_array_dev<float> get_multi_var(const std::string& varname) const;
  multi_array_dev<float> get_multi_mesh(int comp) const;
  multi_array_dev<float> get_quad_var(const std::string& varname, int group,
                                 int rank) const;
  multi_array_dev<float> get_quad_var(const std::string& varname,
                                 int rank) const;
  multi_array_dev<float> get_quad_var(const std::string& varname) const;
  multi_array_dev<float> get_coord_array(int dir) const;
  multi_array_dev<float> get_raw_array(const std::string& varname) const;

  bool find_var(const std::string& varname) const;
  bool is_multimesh() const { return m_isMultimesh; }
  bool is_open() const { return m_open; }
  const std::array<std::string, 3>& grid_conf() const {
    return m_grid_conf;
  }
  const std::string& filename() const { return m_filename; }

 private:
  bool m_open = false;
  bool m_isMultimesh = false;

  DBfile* m_dbfile = nullptr;
  DBtoc* m_dbToc = nullptr;
  DBmultimesh* m_multiMesh = nullptr;
  std::vector<DBquadmesh*> m_dbQuadmeshes;

  std::array<std::string, 3> m_grid_conf;
  std::string m_filename;
  std::string m_parent_dir;

  std::vector<Index> m_pos;
  int m_dims[3];          //!< Dimension of the total grid
  int m_numRanks;         //!< total number of ranks in the output
  int m_domainDecomp[3];  //!< number of domains in each direction
  int m_domainSize[3];    //!< size of each domain in each direction
  int m_guard[3];         //!< number of guard cells in each direction
};                        // ----- end of class silo_file -----

}  // namespace Aperture

#endif  // _SILO_FILE_H_
