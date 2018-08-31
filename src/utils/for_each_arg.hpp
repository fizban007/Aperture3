#ifndef _FOR_EACH_ARG_H_
#define _FOR_EACH_ARG_H_

#include "visit_struct/visit_struct.hpp"

namespace Aperture {

template <int N, typename T, typename U, typename Op>
struct iterate_struct {
  static void run(T& t, U& u, const Op& op) {
    op(visit_struct::get<N>(t), visit_struct::get<N>(u));
    iterate_struct<N-1, T, U, Op>{}.run(t, u, op);
  }
  static void run_with_name(T& t, U& u, const Op& op) {
    op(visit_struct::get_name<N>(t), visit_struct::get<N>(t), visit_struct::get<N>(u));
    iterate_struct<N-1, T, U, Op>{}.run_with_name(t, u, op);
  }
};

template <typename T, typename U, typename Op>
struct iterate_struct<-1, T, U, Op> {
  static void run(T& t, U& u, const Op& op) {}
  static void run_with_name(T& t, U& u, const Op& op) {}
};

template <typename Data, typename U, typename Op>
void for_each_arg(Data& data, U& u, const Op& op) {
  iterate_struct<visit_struct::field_count<Data>()-1, Data, U, Op>::run(data, u, op);
}

template <typename Data, typename U, typename Op>
void for_each_arg_with_name(Data& data, U& u, const Op& op) {
  iterate_struct<visit_struct::field_count<Data>()-1, Data, U, Op>::run_with_name(data, u, op);
}



}

#endif  // _FOR_EACH_ARG_H_
