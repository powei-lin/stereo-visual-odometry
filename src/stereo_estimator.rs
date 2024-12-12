use std::collections::HashMap;

use camera_intrinsic_model::GenericModel;
use image::DynamicImage;
use nalgebra as na;
use patch_tracker::StereoPatchTracker;

use crate::{triangulate::triangulate_points, types::Extrinsics};

pub struct StereoEstimator {
    cam0: GenericModel<f64>,
    cam1: GenericModel<f64>,
    t_1_0: na::Isometry3<f64>,
    t_1_0_mat34: na::SMatrix<f64, 3, 4>,
    t_origin_cam0_current: na::Isometry3<f64>,
    stereo_point_tracker: StereoPatchTracker<5>,
    tracked_points_map: HashMap<usize, na::Vector3<f64>>,
    current_frame_points: (HashMap<usize, (f32, f32)>, HashMap<usize, (f32, f32)>),
}

impl StereoEstimator {
    pub fn new(
        cam0: GenericModel<f64>,
        cam1: GenericModel<f64>,
        extrinsics: &Extrinsics,
    ) -> StereoEstimator {
        let t_1_0 = extrinsics.rtvecs[1].to_na_isometry3();
        let m: na::Matrix4<f64> = t_1_0.to_matrix();
        let t_1_0_mat34 = m.fixed_view::<3, 4>(0, 0);

        StereoEstimator {
            cam0,
            cam1,
            t_1_0,
            t_1_0_mat34: t_1_0_mat34.into(),
            t_origin_cam0_current: na::Isometry3::identity(),
            stereo_point_tracker: StereoPatchTracker::default(),
            tracked_points_map: HashMap::new(),
            current_frame_points: (HashMap::new(), HashMap::new()),
        }
    }
    pub fn process(&mut self, image0: &DynamicImage, image1: &DynamicImage) {
        self.stereo_point_tracker
            .process_frame(&image0.to_luma8(), &image1.to_luma8());
        self.current_frame_points = self.stereo_point_tracker.get_track_points().into();

        // initialized
        if self.tracked_points_map.len() > 0 {
            // compute current cam pose
            let (p3ds, p2ds_z): (Vec<_>, Vec<_>) = self
                .current_frame_points
                .0
                .iter()
                .filter_map(|(i, p)| {
                    if let Some(p3d) = self.tracked_points_map.get(i) {
                        let p2dz = self.cam0.unproject_one(&na::Vector2::new(p.0, p.1).cast());
                        Some(((p3d[0], p3d[1], p3d[2]), (p2dz[0], p2dz[1])))
                    } else {
                        None
                    }
                })
                .unzip();
            let ((rx, ry, rz), (tx, ty, tz)) = sqpnp_simple::sqpnp_solve(&p3ds, &p2ds_z).unwrap();
            let t_cam_world = na::Isometry3::new(na::Vector3::new(tx, ty, tz), na::Vector3::new(rx, ry, rz));
            
            self.t_origin_cam0_current = t_cam_world.inverse();
            
        }

        for (i, pt0) in &self.current_frame_points.0 {
            if self.tracked_points_map.contains_key(i){
                continue;
            }
            if let Some(pt1) = self.current_frame_points.1.get(i) {
                let pt0_undistort = self
                    .cam0
                    .unproject_one(&na::Vector2::new(pt0.0, pt0.1).cast());
                let pt1_undistort = self
                    .cam1
                    .unproject_one(&na::Vector2::new(pt1.0, pt1.1).cast());
                let p3d_current_frame = triangulate_points(&pt0_undistort, &pt1_undistort, &self.t_1_0_mat34);
                if p3d_current_frame[2] < 0.0 || p3d_current_frame[2] > 10.0{
                    continue;
                }
                let p3d_world = self.t_origin_cam0_current * p3d_current_frame;
                self.tracked_points_map.insert(
                    *i,
                    p3d_world,
                );
            }
        }
    }
    pub fn get_current_frame_points(
        &self,
    ) -> (&HashMap<usize, (f32, f32)>, &HashMap<usize, (f32, f32)>) {
        (&self.current_frame_points.0, &self.current_frame_points.1)
    }
    pub fn get_track_points(&self) -> &HashMap<usize, na::Vector3<f64>> {
        &self.tracked_points_map
    }
    pub fn get_current_cam0_pose(&self) -> &na::Isometry3<f64>{
        &self.t_origin_cam0_current
    }
}
