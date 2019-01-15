#ifndef _FILESYSTEM_H_
#define _FILESYSTEM_H_

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

namespace Aperture {

bool
copyDir(boost::filesystem::path const &source,
        boost::filesystem::path const &destination);

}

#endif  // _FILESYSTEM_H_
