#ifndef _FILESYSTEM_H_
#define _FILESYSTEM_H_

#include <string>

#if __GNUC__ >= 8 || __clang_major__ >= 7
#include <filesystem>
#undef USE_BOOST_FILESYSTEM
#else
#define USE_BOOST_FILESYSTEM
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#endif

namespace Aperture {

#ifdef USE_BOOST_FILESYSTEM
namespace fs = boost::filesystem;
using Path = boost::filesystem::path;
#else
namespace fs = std::filesystem;
using Path = std::filesystem::path;
#endif

void
copy_file(const std::string& src, const std::string& dest);



}

#endif  // _FILESYSTEM_H_
