#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "irl/generic_cutting/generic_cutting.h"
#include "irl/generic_cutting/paraboloid_intersection/paraboloid_intersection.h"
#include "irl/geometry/general/normal.h"
#include "irl/geometry/general/pt.h"
#include "irl/geometry/general/reference_frame.h"
#include "irl/geometry/polyhedrons/tet.h"
#include "irl/moments/volume_moments.h"
#include "irl/paraboloid_reconstruction/paraboloid.h"

namespace {

struct TetIds {
  int ids[4];
};

double signed_tet_volume(const IRL::Pt& a, const IRL::Pt& b, const IRL::Pt& c,
                         const IRL::Pt& d) {
  const auto ax = a[0] - d[0];
  const auto ay = a[1] - d[1];
  const auto az = a[2] - d[2];
  const auto bx = b[0] - d[0];
  const auto by = b[1] - d[1];
  const auto bz = b[2] - d[2];
  const auto cx = c[0] - d[0];
  const auto cy = c[1] - d[1];
  const auto cz = c[2] - d[2];
  return (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) +
          az * (bx * cy - by * cx)) /
         6.0;
}

IRL::Tet make_positive_tet(std::array<IRL::Pt, 4> pts) {
  if (signed_tet_volume(pts[0], pts[1], pts[2], pts[3]) < 0.0) {
    std::swap(pts[0], pts[1]);
  }
  return IRL::Tet({pts[0], pts[1], pts[2], pts[3]});
}

}  // namespace

int main() {
  try {
    int npts = 0;
    int ntets = 0;
    if (!(std::cin >> npts >> ntets)) {
      throw std::runtime_error("Expected npts ntets.");
    }

    double dx = 0.0, dy = 0.0, dz = 0.0;
    std::cin >> dx >> dy >> dz;

    double f00 = 0.0, f01 = 0.0, f02 = 0.0;
    double f10 = 0.0, f11 = 0.0, f12 = 0.0;
    double f20 = 0.0, f21 = 0.0, f22 = 0.0;
    std::cin >> f00 >> f01 >> f02 >> f10 >> f11 >> f12 >> f20 >> f21 >> f22;

    double coef_a = 0.0, coef_b = 0.0;
    int use_above_region = 0;
    std::cin >> coef_a >> coef_b >> use_above_region;

    std::vector<IRL::Pt> points;
    points.reserve(static_cast<std::size_t>(npts));
    for (int i = 0; i < npts; ++i) {
      double x = 0.0, y = 0.0, z = 0.0;
      std::cin >> x >> y >> z;
      points.emplace_back(x, y, z);
    }

    std::vector<TetIds> tets;
    tets.reserve(static_cast<std::size_t>(ntets));
    for (int i = 0; i < ntets; ++i) {
      TetIds tet{};
      std::cin >> tet.ids[0] >> tet.ids[1] >> tet.ids[2] >> tet.ids[3];
      tets.push_back(tet);
    }

    const IRL::ReferenceFrame frame(
        IRL::Normal(f00, f01, f02), IRL::Normal(f10, f11, f12),
        IRL::Normal(f20, f21, f22));
    const IRL::Paraboloid paraboloid(IRL::Pt(dx, dy, dz), frame, coef_a,
                                     coef_b);

    double total_volume = 0.0;
    double below_volume = 0.0;
    int zero_tets = 0;

    for (const auto& ids : tets) {
      std::array<IRL::Pt, 4> tet_pts = {
          points.at(static_cast<std::size_t>(ids.ids[0])),
          points.at(static_cast<std::size_t>(ids.ids[1])),
          points.at(static_cast<std::size_t>(ids.ids[2])),
          points.at(static_cast<std::size_t>(ids.ids[3]))};
      if (std::fabs(signed_tet_volume(tet_pts[0], tet_pts[1], tet_pts[2],
                                      tet_pts[3])) < 1.0e-18) {
        ++zero_tets;
        continue;
      }
      const auto tet = make_positive_tet(tet_pts);
      const double tet_volume = std::fabs(static_cast<double>(tet.calculateVolume()));
      const auto moments = IRL::getVolumeMoments<IRL::VolumeMoments>(tet, paraboloid);
      double clipped_below = static_cast<double>(moments.volume());
      if (clipped_below < 0.0 && clipped_below > -1.0e-13) {
        clipped_below = 0.0;
      }
      if (clipped_below > tet_volume && clipped_below - tet_volume < 1.0e-13) {
        clipped_below = tet_volume;
      }
      total_volume += tet_volume;
      below_volume += clipped_below;
    }

    const double selected_volume =
        use_above_region ? total_volume - below_volume : below_volume;
    std::cout << std::setprecision(17);
    std::cout << "total_volume " << total_volume << "\n";
    std::cout << "below_volume " << below_volume << "\n";
    std::cout << "above_volume " << total_volume - below_volume << "\n";
    std::cout << "selected_volume " << selected_volume << "\n";
    std::cout << "ntets " << ntets << "\n";
    std::cout << "zero_tets " << zero_tets << "\n";
  } catch (const std::exception& err) {
    std::cerr << "irl_paraboloid_clip_volume: " << err.what() << "\n";
    return 1;
  }

  return 0;
}
