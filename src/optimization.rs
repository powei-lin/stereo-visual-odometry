use std::collections::HashMap;

use nalgebra as na;
use num_dual::DualDVec64;
use tiny_solver::{factors::Factor, loss_functions::HuberLoss, Optimizer};

use crate::types::{DVecVec3, ToRvecTvec};

#[derive(Debug, Clone)]
pub struct PnPFactor<T: na::RealField> {
    t_i_0: na::Isometry3<T>,
    p3d: na::Point3<T>,
    p2d: na::Point2<T>,
}

impl<T: na::RealField> PnPFactor<T> {
    pub fn new(
        t_i_0: &na::Isometry3<f64>,
        p3d: &(f64, f64, f64),
        undist_p2d: &(f64, f64),
    ) -> PnPFactor<T> {
        PnPFactor {
            t_i_0: t_i_0.cast(),
            p3d: na::Point3::new(p3d.0, p3d.1, p3d.2).cast(),
            p2d: na::Point2::new(undist_p2d.0, undist_p2d.1).cast(),
        }
    }
    pub fn reprojection_error(&self, t_cam0_world: &na::Isometry3<T>) -> na::DVector<T> {
        let p3d_t = self.t_i_0.clone() * t_cam0_world * self.p3d.clone();
        na::dvector![
            self.p2d.x.clone() - p3d_t.x.clone() / p3d_t.z.clone(),
            self.p2d.y.clone() - p3d_t.y.clone() / p3d_t.z.clone()
        ]
    }
}

impl Factor for PnPFactor<DualDVec64> {
    fn residual_func(
        &self,
        params: &[nalgebra::DVector<num_dual::DualDVec64>],
    ) -> nalgebra::DVector<num_dual::DualDVec64> {
        let rvec = na::Vector3::new(
            params[0][0].clone(),
            params[0][1].clone(),
            params[0][2].clone(),
        );
        let tvec = na::Vector3::new(
            params[1][0].clone(),
            params[1][1].clone(),
            params[1][2].clone(),
        );
        let t_cam0_world = na::Isometry3::new(tvec, rvec);
        self.reprojection_error(&t_cam0_world)
    }
}

pub fn pnp_refine(
    t_1_0: &na::Isometry3<f64>,
    t_cam0_origin_init: &na::Isometry3<f64>,
    cam0_pts: &[(usize, (f64, f64, f64), (f64, f64))],
    cam1_pts: &[(usize, (f64, f64, f64), (f64, f64))],
) -> Option<(na::Isometry3<f64>, Vec<usize>)> {
    let rtvec = t_cam0_origin_init.to_rvec_tvec();
    let mut problem = tiny_solver::Problem::new();

    let mut cam0_cost = HashMap::new();
    let mut cam1_cost = HashMap::new();
    for (id, p3d, p2d) in cam0_pts {
        let cost = PnPFactor::new(&na::Isometry3::identity(), p3d, p2d);
        cam0_cost.insert(*id, cost.clone());
        problem.add_residual_block(
            2,
            vec![("rvec".to_string(), 3), ("tvec".to_string(), 3)],
            Box::new(cost),
            Some(Box::new(HuberLoss::new(0.5))),
        );
    }
    for (id, p3d, p2d) in cam1_pts {
        let cost = PnPFactor::new(t_1_0, p3d, p2d);
        cam1_cost.insert(*id, cost.clone());
        problem.add_residual_block(
            2,
            vec![("rvec".to_string(), 3), ("tvec".to_string(), 3)],
            Box::new(cost),
            Some(Box::new(HuberLoss::new(0.1))),
        );
    }
    let initial_values = HashMap::<String, na::DVector<f64>>::from([
        ("rvec".to_string(), rtvec.na_rvec()),
        ("tvec".to_string(), rtvec.na_tvec()),
    ]);
    let optimizer = tiny_solver::GaussNewtonOptimizer {};
    let threshold = 0.1;

    if let Some(result) = optimizer.optimize(&problem, &initial_values, None) {
        let t_cam_world = na::Isometry3::new(
            result.get("tvec").unwrap().to_vec3(),
            result.get("rvec").unwrap().to_vec3(),
        );

        let mut good_id_errs = Vec::new();
        let mut bad_ids = Vec::new();
        for (id0, p3d, p2d) in cam0_pts {
            let cost = PnPFactor::<f64>::new(&na::Isometry3::identity(), p3d, p2d);
            let rep = cost.reprojection_error(&t_cam_world);
            let rep_sq = rep.norm_squared();
            if rep_sq > threshold {
                bad_ids.push(*id0);
            } else {
                good_id_errs.push((*id0, rep_sq));
            }
        }
        for (id1, p3d, p2d) in cam1_pts {
            let cost = PnPFactor::<f64>::new(&na::Isometry3::identity(), p3d, p2d);
            let rep = cost.reprojection_error(&t_cam_world);
            let rep_sq = rep.norm_squared();
            if rep_sq > threshold {
                bad_ids.push(*id1);
            } else {
                good_id_errs.push((*id1, rep_sq));
            }
        }
        good_id_errs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        good_id_errs.truncate(30);

        let mut refine_problem = tiny_solver::Problem::new();

        for (id, _) in good_id_errs {
            if let Some(cost) = cam0_cost.remove(&id) {
                refine_problem.add_residual_block(
                    2,
                    vec![("rvec".to_string(), 3), ("tvec".to_string(), 3)],
                    Box::new(cost),
                    Some(Box::new(HuberLoss::new(0.5))),
                );
            }
            if let Some(cost) = cam1_cost.remove(&id) {
                refine_problem.add_residual_block(
                    2,
                    vec![("rvec".to_string(), 3), ("tvec".to_string(), 3)],
                    Box::new(cost),
                    Some(Box::new(HuberLoss::new(0.5))),
                );
            }
        }
        let result = optimizer.optimize(&refine_problem, &result, None).unwrap();
        let t_cam_world = na::Isometry3::new(
            result.get("tvec").unwrap().to_vec3(),
            result.get("rvec").unwrap().to_vec3(),
        );

        return Some((t_cam_world, bad_ids));
    }

    None
}