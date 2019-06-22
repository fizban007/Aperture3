#include "grid_log_sph_ptrs.h"

namespace Aperture {

static mesh_ptrs_log_sph g_mesh_ptrs;
static bool g_mesh_ptrs_initialized = false;

mesh_ptrs_log_sph
get_mesh_ptrs(Grid_LogSph& grid) {
  if (!g_mesh_ptrs_initialized) {
    g_mesh_ptrs.l1_e = get_pitchptr(grid.m_l1_e);
    g_mesh_ptrs.l2_e = get_pitchptr(grid.m_l2_e);
    g_mesh_ptrs.l3_e = get_pitchptr(grid.m_l3_e);
    g_mesh_ptrs.l1_b = get_pitchptr(grid.m_l1_b);
    g_mesh_ptrs.l2_b = get_pitchptr(grid.m_l2_b);
    g_mesh_ptrs.l3_b = get_pitchptr(grid.m_l3_b);

    g_mesh_ptrs.A1_e = get_pitchptr(grid.m_A1_e);
    g_mesh_ptrs.A2_e = get_pitchptr(grid.m_A2_e);
    g_mesh_ptrs.A3_e = get_pitchptr(grid.m_A3_e);
    g_mesh_ptrs.A1_b = get_pitchptr(grid.m_A1_b);
    g_mesh_ptrs.A2_b = get_pitchptr(grid.m_A2_b);
    g_mesh_ptrs.A3_b = get_pitchptr(grid.m_A3_b);

    g_mesh_ptrs.dV = get_pitchptr(grid.m_dV);

    g_mesh_ptrs_initialized = true;
  }
  return g_mesh_ptrs;
}
}
