use std::collections::{HashMap, HashSet};

use camera_intrinsic_model::GenericModel;
use image::DynamicImage;
use nalgebra::{self as na, Quaternion};
use patch_tracker::StereoPatchTracker;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{optimization::pnp_refine, triangulate::triangulate_points, types::Extrinsics};

pub struct StereoEstimator {
    cam0: GenericModel<f64>,
    cam1: GenericModel<f64>,
    t_1_0: na::Isometry3<f64>,
    t_1_0_mat34: na::SMatrix<f64, 3, 4>,
    t_cam0_origin_current: na::Isometry3<f64>,
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
        let t_cam0_origin_current =
            na::Isometry3::new(na::Vector3::zeros(), na::Vector3::new(0.0, 0.0, 1e-6));
        StereoEstimator {
            cam0,
            cam1,
            t_1_0,
            t_1_0_mat34: t_1_0_mat34.into(),
            t_cam0_origin_current: t_cam0_origin_current,
            stereo_point_tracker: StereoPatchTracker::default(),
            tracked_points_map: HashMap::new(),
            current_frame_points: (HashMap::new(), HashMap::new()),
        }
    }
    pub fn process(&mut self, image0: &DynamicImage, image1: &DynamicImage) {
        self.stereo_point_tracker
            .process_frame(&image0.to_luma8(), &image1.to_luma8());
        self.current_frame_points = self.stereo_point_tracker.get_track_points().into();
        let tracked_id_cam0: HashSet<usize> = self
            .current_frame_points
            .0
            .keys()
            .filter(|&k| self.tracked_points_map.contains_key(k))
            .map(|&k| k)
            .collect();
        let tracked_id_cam1: HashSet<usize> = self
            .current_frame_points
            .1
            .keys()
            .filter(|&k| self.tracked_points_map.contains_key(k))
            .map(|&k| k)
            .collect();
        let untracked_both: HashSet<usize> = self
            .current_frame_points
            .0
            .keys()
            .filter(|&k| !self.tracked_points_map.contains_key(k))
            .map(|&k| k)
            .collect();

        // initialized
        if self.tracked_points_map.len() > 0 {
            // compute current cam pose
            let cam0_pts: Vec<_> = tracked_id_cam0
                .iter()
                .map(|i| {
                    let p3d = self.tracked_points_map.get(i).unwrap();
                    let p2d = self.current_frame_points.0.get(i).unwrap();
                    let p2dz = self
                        .cam0
                        .unproject_one(&na::Vector2::new(p2d.0, p2d.1).cast());
                    (*i, (p3d[0], p3d[1], p3d[2]), (p2dz[0], p2dz[1]))
                })
                .collect();
            let cam1_pts: Vec<_> = tracked_id_cam1
                .iter()
                .map(|i| {
                    let p3d = self.tracked_points_map.get(i).unwrap();
                    let p2d = self.current_frame_points.1.get(i).unwrap();
                    let p2dz = self
                        .cam0
                        .unproject_one(&na::Vector2::new(p2d.0, p2d.1).cast());
                    (*i, (p3d[0], p3d[1], p3d[2]), (p2dz[0], p2dz[1]))
                })
                .collect();
            let (p3ds0, p2ds0_z): (Vec<_>, Vec<_>) = cam0_pts.iter().map(|i| (i.1, i.2)).unzip();
            // let ((rx, ry, rz), (tx, ty, tz)) = sqpnp_simple::sqpnp_solve(&p3ds0, &p2ds0_z).unwrap();
            // let initial_rtvec = sqpnp_simple::sqpnp_solve(&p3ds0, &p2ds0_z).unwrap();
            if let Some((t_cam_world, bad_ids)) = pnp_refine(
                &self.t_1_0,
                &self.t_cam0_origin_current,
                &cam0_pts,
                &cam1_pts,
            ) {
                self.t_cam0_origin_current = t_cam_world;
                // self.stereo_point_tracker.remove_id(&bad_ids);
                // self.tracked_points_map.remove();
            } else {
                println!("failed");
            }
        }
        let t_origin_cam0 = self.t_cam0_origin_current.inverse();
        for (i, pt0) in &self.current_frame_points.0 {
            if self.tracked_points_map.contains_key(i) {
                continue;
            }
            if let Some(pt1) = self.current_frame_points.1.get(i) {
                let pt0_undistort = self
                    .cam0
                    .unproject_one(&na::Vector2::new(pt0.0, pt0.1).cast());
                let pt1_undistort = self
                    .cam1
                    .unproject_one(&na::Vector2::new(pt1.0, pt1.1).cast());
                let p3d_current_frame =
                    triangulate_points(&pt0_undistort, &pt1_undistort, &self.t_1_0_mat34);
                if p3d_current_frame[2] < 0.0 || p3d_current_frame[2] > 5.0 {
                    continue;
                }
                let p3d_world = t_origin_cam0 * p3d_current_frame;
                self.tracked_points_map.insert(*i, p3d_world);
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
    pub fn get_current_cam0_pose(&self) -> &na::Isometry3<f64> {
        &self.t_cam0_origin_current
    }
}
