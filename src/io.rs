use std::path::PathBuf;

use crate::types::Extrinsics;
use glob::glob;
use image::{DynamicImage, ImageReader};

pub fn extrinsics_from_json(file_path: &str) -> Extrinsics {
    let contents =
        std::fs::read_to_string(file_path).expect("Should have been able to read the file");
    serde_json::from_str(&contents).unwrap()
}

fn path_to_timestamp(path: &PathBuf) -> i64 {
    let time_ns: i64 = path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap_or(0);
    time_ns
}

pub struct EurocImageLoader {
    // two cameras
    image_paths: [Vec<PathBuf>; 2],
}

impl EurocImageLoader {
    pub fn new(folder_path: &str) -> EurocImageLoader {
        let img_paths: Vec<_> = (0..2)
            .map(|cam_idx| {
                let img_paths =
                    glob(format!("{}/mav0/cam{}/data/*.png", folder_path, cam_idx).as_str())
                        .expect("failed");
                let mut sorted_path: Vec<_> = img_paths.collect();
                // reverse time
                sorted_path.sort_by(|a, b| b.as_ref().unwrap().cmp(a.as_ref().unwrap()));
                let paths: Vec<_> = sorted_path
                    .iter()
                    .map(|f| f.as_ref().unwrap().to_owned())
                    .collect();
                paths
            })
            .collect();
        if img_paths[0].len() != img_paths[1].len() {
            panic!("cam0 and cam1 need to have the same amount of images.")
        }
        EurocImageLoader {
            image_paths: [img_paths[0].clone(), img_paths[1].clone()],
        }
    }
}

impl Iterator for EurocImageLoader {
    type Item = (i64, DynamicImage, DynamicImage);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p0) = self.image_paths[0].pop() {
            let p1 = self.image_paths[1].pop().unwrap();
            let time_ns = path_to_timestamp(&p0);
            let img0 = ImageReader::open(p0).unwrap().decode().unwrap();
            let img1 = ImageReader::open(p1).unwrap().decode().unwrap();
            Some((time_ns, img0, img1))
        } else {
            None
        }
    }
}
