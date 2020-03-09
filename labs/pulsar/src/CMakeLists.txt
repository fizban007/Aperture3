set(LAB_SHADOWS "")
set(LAB_NAME "pulsar")
# list(APPEND LAB_CU_SHADOWS "cuda/core/ptc_updater_logsph.cu" "cuda/radiation/rt_pulsar.cu" "cuda/core/field_solver_log_sph.cu" "cuda/core/cu_sim_data.cu")
list(APPEND LAB_SHADOWS "cuda/sim_data.cu")
