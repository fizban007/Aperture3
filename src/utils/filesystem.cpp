#include "utils/filesystem.h"
#include <iostream>

namespace Aperture {

void copy_file(const std::string& src, const std::string& dest) {
#ifndef USE_BOOST_FILESYSTEM
    fs::copy_file(src, dest, fs::copy_options::overwrite_existing);
#else
    fs::copy_file(src, dest, fs::copy_option::overwrite_if_exists);
#endif
}

// bool
// copyDir(boost::filesystem::path const &source,
//         boost::filesystem::path const &destination) {
//   namespace fs = boost::filesystem;
//   try {
//     // Check whether the function call is valid
//     if (!fs::exists(source) || !fs::is_directory(source)) {
//       std::cerr << "Source directory " << source.string()
//                 << " does not exist or is not a directory." << '\n';
//       return false;
//     }
//     // if (fs::exists(destination)) {
//     //   std::cerr << "Destination directory " << destination.string()
//     //             << " already exists." << '\n';
//     //   return false;
//     // }
//     // Create the destination directory
//     if (!fs::exists(destination)) {
//       if (!fs::create_directory(destination)) {
//         std::cerr << "Unable to create destination directory"
//                   << destination.string() << '\n';
//         return false;
//       }
//     }
//   } catch (fs::filesystem_error const &e) {
//     std::cerr << e.what() << '\n';
//     return false;
//   }
//   // Iterate through the source directory
//   for (fs::directory_iterator file(source);
//        file != fs::directory_iterator(); ++file) {
//     try {
//       fs::path current(file->path());
//       if (fs::is_directory(current)) {
//         // Found directory: Recursion
//         if (!copyDir(current, destination / current.filename())) {
//           return false;
//         }
//       } else {
//         // Found file: Copy
//         fs::copy_file(
//             current, destination / current.filename(),
//             boost::filesystem::copy_option::overwrite_if_exists);
//       }
//     } catch (fs::filesystem_error const &e) {
//       std::cerr << e.what() << '\n';
//     }
//   }
//   return true;
// }

} // namespace Aperture
