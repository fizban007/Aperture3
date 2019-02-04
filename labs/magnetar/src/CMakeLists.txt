set(LAB_SHADOWS "")
set(LAB_NAME "magnetar")
#list(APPEND LAB_SHADOWS "main.cpp")
list(APPEND LAB_CU_SHADOWS "cuda/core/ptc_updater_logsph.cu" "cuda/radiation/rt_magnetar.cu")
