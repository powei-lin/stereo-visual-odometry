use nalgebra as na;
use num_dual::DualDVec64;
use tiny_solver::factors::Factor;

pub struct PnPFactor<T: na::RealField>{
    t_i_0: na::Isometry3<T>,
    p3d: na::Point3<T>,
    p2d: na::Point2<T>
}

impl <T: na::RealField> PnPFactor<T> {
    pub fn new(t_i_0: &na::Isometry3<f64>, p3d: &(f64, f64, f64), undist_p2d: &(f64, f64)) -> PnPFactor<T>{
        PnPFactor{
            t_i_0: t_i_0.cast(),
            p3d: na::Point3::new(p3d.0, p3d.1, p3d.2).cast(),
            p2d: na::Point2::new(undist_p2d.0, undist_p2d.1).cast(),
        }
    }
    pub fn reprojection_error(&self, t_cam0_world: &na::Isometry3<T>) -> na::DVector<T>{
        let p3d_t = self.t_i_0.clone() * t_cam0_world * self.p3d.clone();
        na::dvector![self.p2d.x.clone() - p3d_t.x.clone() / p3d_t.z.clone(), self.p2d.y.clone() - p3d_t.y.clone() / p3d_t.z.clone()]
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

pub fn pnp_refine(t_cam0_world_init: &na::Isometry3<f64>, ){
    
}