#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <string>
#include <cstdio>
#include <fmt/ostream.h>
#include "data/enum_types.h"

namespace Aperture {

class Logger {
 private:
  static int m_rank;
  static LogLevel m_level;
  static std::string m_log_file;
  static std::FILE* m_file;

 public:
  Logger() {}
  ~Logger();

  static void init(int rank, LogLevel level, std::string log_file);

  template <typename... Args>
  static void err(const char* str, Args&&... args) {
    fmt::print(stderr, str, std::forward<Args>(args)...);
    fmt::print("\n");
  }

  template <typename... Args>
  static void print_err(const char* str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(stderr, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_info(const char* str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_detail(const char* str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::info) {
      fmt::print(str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_debug(const char* str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::detail) {
      fmt::print(str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_debug_all(const char* str, Args&&... args) {
    if (m_level > LogLevel::detail) {
      fmt::print(str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_info(const char* str, Args&&... args) {
    if (m_file == nullptr) return;
    if (m_rank == 0) {
      fmt::print(m_file, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_detail(const char* str, Args&&... args) {
    if (m_file == nullptr) return;
    if (m_rank == 0 && m_level > LogLevel::info) {
      fmt::print(m_file, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_debug(const char* str, Args&&... args) {
    if (m_file == nullptr) return;
    if (m_rank == 0 && m_level > LogLevel::detail) {
      fmt::print(m_file, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_debug_all(const char* str, Args&&... args) {
    if (m_file == nullptr) return;
    if (m_level > LogLevel::detail) {
      fmt::print(m_file, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }


};

}

#endif  // _LOGGER_H_
