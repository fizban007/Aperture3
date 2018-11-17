#ifndef _TYPE_NAME_H_
#define _TYPE_NAME_H_

#include <typeinfo>

namespace Aperture {

// default implementation
template <typename T>
struct TypeName
{
    static const char* Get()
    {
        return typeid(T).name();
    }
};

// a specialization for each type of those you want to support
// and don't like the string returned by typeid
template <>
struct TypeName<float>
{
    static const char* Get()
    {
        return "float";
    }
};

template <>
struct TypeName<double>
{
    static const char* Get()
    {
        return "double";
    }
};

}

#endif  // _TYPE_NAME_H_
