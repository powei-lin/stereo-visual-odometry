use nalgebra as na;

// Eigen::Matrix<Scalar_2, 3, 4> P1, P2;
// P1.setIdentity();
// P2 = T_0_1.inverse().matrix3x4();

// Eigen::Matrix<Scalar_2, 4, 4> A(4, 4);
// A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
// A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
// A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
// A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);

// Eigen::JacobiSVD<Eigen::Matrix<Scalar_2, 4, 4>> mySVD(A, Eigen::ComputeFullV);
// Vec4_2 worldPoint = mySVD.matrixV().col(3);
// worldPoint /= worldPoint.template head<3>().norm();

// // Enforce same direction of bearing vector and initial point
// if (f0.dot(worldPoint.template head<3>()) < 0)
//   worldPoint *= -1;

// return worldPoint;

pub fn triangulate_points(
    undist_pt0: &na::Vector3<f64>,
    undist_pt1: &na::Vector3<f64>,
    t_1_0: &na::SMatrix<f64, 3, 4>,
) -> na::Vector3<f64> {
    // println!("{}", t_1_0);
    let r0 = undist_pt1[0] * t_1_0.row(2) - t_1_0.row(0);
    let r1 = undist_pt1[1] * t_1_0.row(2) - t_1_0.row(1);
    let design_matrix = unsafe {
        na::Matrix4::new(
            -1.0,
            0.0,
            undist_pt0[0],
            0.0,
            0.0,
            -1.0,
            undist_pt0[1],
            0.0,
            *r0.get_unchecked(0),
            *r0.get_unchecked(1),
            *r0.get_unchecked(2),
            *r0.get_unchecked(3),
            *r1.get_unchecked(0),
            *r1.get_unchecked(1),
            *r1.get_unchecked(2),
            *r1.get_unchecked(3),
        )
    };
    let svd = design_matrix.svd(false, true);
    let vt: na::Matrix4<f64> = svd.v_t.unwrap();
    let p3d = vt.row(3) / vt[(3, 3)];
    p3d.transpose().fixed_rows::<3>(0).into()
}
