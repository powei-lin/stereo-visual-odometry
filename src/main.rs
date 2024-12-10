use glob::glob;
use image::{DynamicImage, ImageReader};
use patch_tracker::StereoPatchTracker;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::io::Cursor;

fn get_tum_vi_stereo_paths(dataset_root: &str) -> Vec<(u64, String, String)> {
    let cam0_paths =
        glob(format!("{}/mav0/cam0/data/*.png", dataset_root).as_str()).expect("cam0 failed");
    let cam1_paths =
        glob(format!("{}/mav0/cam1/data/*.png", dataset_root).as_str()).expect("cam1 failed");
    cam0_paths
        .zip(cam1_paths)
        .filter_map(|(p0, p1)| {
            if let Ok(p0) = p0 {
                if let Ok(p1) = p1 {
                    let ts: u64 = p0
                        .as_path()
                        .file_stem()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .parse()
                        .unwrap();
                    return Some((
                        ts,
                        p0.to_str().unwrap().to_owned(),
                        p1.to_str().unwrap().to_owned(),
                    ));
                }
            }
            None
        })
        .collect()
}

fn log_image_as_jpeg(recording: &RecordingStream, topic: &str, img: &DynamicImage) {
    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)
        .unwrap();
    recording
        .log(
            format!("{}/image", topic),
            &rerun::Image::from_file_contents(bytes, None).unwrap(),
        )
        .unwrap();
}
type VecPtsColors = Vec<(Vec<(f32, f32)>, Vec<(u8, u8, u8, u8)>)>;

fn get_p2ds_and_colors<const N: u32>(point_tracker: &StereoPatchTracker<N>) -> VecPtsColors {
    let tracked_stereo_pts = point_tracker.get_track_points();
    tracked_stereo_pts
        .iter()
        .map(|tracked_pts| {
            tracked_pts
                .iter()
                .map(|(id, (x, y))| {
                    let mut rng = ChaCha8Rng::seed_from_u64(*id as u64);
                    let color_num = rng.gen_range(0..2u32.pow(24));
                    let color = (
                        ((color_num >> 16) % 256) as u8,
                        ((color_num >> 8) % 256) as u8,
                        (color_num % 256) as u8,
                        255,
                    );

                    ((*x, *y), color)
                })
                .unzip()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("visual odometry").spawn()?;
    let dataset_root = "/Users/powei/Documents/dataset/tum_vi/dataset-corridor4_512_16";
    let dataset_path_tuple_list = get_tum_vi_stereo_paths(dataset_root);

    let mut stereo_point_tracker = StereoPatchTracker::<5>::default();
    for (ts, p0, p1) in dataset_path_tuple_list {
        let img0 = ImageReader::open(&p0)?.decode()?;
        let img0_luma8 = img0.to_luma8();

        let img1 = ImageReader::open(&p1)?.decode()?;
        let img1_luma8 = img1.to_luma8();
        stereo_point_tracker.process_frame(&img0_luma8, &img1_luma8);

        let img0_grey = DynamicImage::ImageLuma8(img0_luma8);
        let img1_grey = DynamicImage::ImageLuma8(img1.to_luma8());
        recording.set_time_nanos("stable_time", ts as i64);
        log_image_as_jpeg(&recording, "/cam0", &img0_grey);
        log_image_as_jpeg(&recording, "/cam1", &img1_grey);

        let p2ds_colors = get_p2ds_and_colors(&stereo_point_tracker);

        for (i, (p2ds, colors)) in p2ds_colors.iter().enumerate() {
            recording
                .log(
                    format!("/cam{}/points", i),
                    &rerun::Points2D::new(p2ds)
                        .with_colors(colors.clone())
                        .with_radii([rerun::Radius::new_scene_units(3.0)]),
                )
                .expect("msg");
        }
    }

    Ok(())
}
