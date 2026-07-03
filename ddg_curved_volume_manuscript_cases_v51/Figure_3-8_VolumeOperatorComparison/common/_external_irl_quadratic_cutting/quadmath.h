#ifndef IRL_LOCAL_QUADMATH_COMPAT_H_
#define IRL_LOCAL_QUADMATH_COMPAT_H_

#include <cfloat>
#include <cmath>
#include <cstdio>

#ifndef __float128
#define __float128 long double
#endif

#ifndef FLT128_MIN
#define FLT128_MIN LDBL_MIN
#endif

#ifndef FLT128_EPSILON
#define FLT128_EPSILON LDBL_EPSILON
#endif

#ifndef M_PIq
#define M_PIq 3.141592653589793238462643383279502884L
#endif

inline int quadmath_snprintf(char* str, size_t size, const char*, long double value) {
  return std::snprintf(str, size, "%+.20Le", value);
}

inline int isnanq(long double x) { return std::isnan(x) ? 1 : 0; }
inline long double fabsq(long double x) { return std::fabs(x); }
inline long double sqrtq(long double x) { return std::sqrt(x); }
inline long double powq(long double x, long double y) { return std::pow(x, y); }
inline long double logq(long double x) { return std::log(x); }
inline long double atanq(long double x) { return std::atan(x); }
inline long double atan2q(long double y, long double x) { return std::atan2(y, x); }
inline long double atanhq(long double x) { return std::atanh(x); }
inline long double sinq(long double x) { return std::sin(x); }
inline long double cosq(long double x) { return std::cos(x); }

#endif
