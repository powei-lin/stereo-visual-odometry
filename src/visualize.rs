use std::io::Cursor;

use image::DynamicImage;
use rerun::RecordingStream;
use nalgebra as na;

pub fn log_image_as_compressed(
    recording: &RecordingStream,
    topic: &str,
    img: &DynamicImage,
    format: image::ImageFormat,
) {
    let mut bytes: Vec<u8> = Vec::new();

    img.to_luma8()
        .write_to(&mut Cursor::new(&mut bytes), format)
        .unwrap();

    recording
        .log(
            format!("{}/image", topic),
            &rerun::Image::from_file_contents(bytes, None).unwrap(),
        )
        .unwrap();
}

/// rerun use top left corner as (0, 0)
pub fn rerun_shift(p2ds: &[(f32, f32)]) -> Vec<(f32, f32)> {
    p2ds.iter().map(|(x, y)| (*x + 0.5, *y + 0.5)).collect()
}

pub fn na_isometry3_to_rerun_transform3d(transform: &na::Isometry3<f64>) -> rerun::Transform3D {
    let t = (
        transform.translation.x as f32,
        transform.translation.y as f32,
        transform.translation.z as f32,
    );
    let q_xyzw = (
        transform.rotation.quaternion().i as f32,
        transform.rotation.quaternion().j as f32,
        transform.rotation.quaternion().k as f32,
        transform.rotation.quaternion().w as f32,
    );
    rerun::Transform3D::from_translation_rotation(t, rerun::Quaternion::from_xyzw(q_xyzw.into()))
}