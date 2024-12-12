use camera_intrinsic_model::io::model_from_json;
use clap::Parser;
use stereo_vo::{
    io::{extrinsics_from_json, EurocImageLoader},
    stereo_estimator::StereoEstimator,
    visualize::{self, log_image_as_compressed, rerun_shift},
};

#[derive(Parser)]
#[command(version, about, author)]
struct StereoVOCli {
    /// path to cam0 cam1 and extrinsic folder
    #[arg(short, long)]
    params_folder: String,

    /// path to euroc format folder
    #[arg(short, long)]
    dataset: String,
}

fn main() {
    env_logger::init();
    let cli = StereoVOCli::parse();
    let cam0_model = model_from_json(&format!("{}/cam0.json", cli.params_folder));
    let cam1_model = model_from_json(&format!("{}/cam1.json", cli.params_folder));
    let extrinsics = extrinsics_from_json(&format!("{}/extrinsics.json", cli.params_folder));
    let mut stereo_esimator = StereoEstimator::new(cam0_model, cam1_model, &extrinsics);
    let dataset_loader = EurocImageLoader::new(&cli.dataset);
    let recording = rerun::RecordingStreamBuilder::new("calibration")
        .save(format!("stereo.rrd"))
        .unwrap();
    recording
        .log_static("/", &rerun::Transform3D::IDENTITY)
        .unwrap();
    recording
        .log_static("/", &rerun::ViewCoordinates::RDF)
        .unwrap();
    for (ts, img0, img1) in dataset_loader {
        println!("ts: {}", ts);
        recording.set_time_nanos("stable_time", ts);
        log_image_as_compressed(&recording, "/cam0", &img0, image::ImageFormat::Jpeg);
        log_image_as_compressed(&recording, "/cam1", &img1, image::ImageFormat::Jpeg);
        stereo_esimator.process(&img0, &img1);
        let (current_points_cam0, current_points_cam1) = stereo_esimator.get_current_frame_points();
        let pts0: Vec<_> = current_points_cam0
            .values()
            .into_iter()
            .map(|p| *p)
            .collect();
        let pts1: Vec<_> = current_points_cam1
            .values()
            .into_iter()
            .map(|p| *p)
            .collect();
        recording
            .log("/cam0/points", &rerun::Points2D::new(rerun_shift(&pts0)))
            .unwrap();
        recording
            .log("/cam1/points", &rerun::Points2D::new(rerun_shift(&pts1)))
            .unwrap();
        let tracked_pts: Vec<_> = stereo_esimator
            .get_track_points()
            .values()
            .map(|f| {
                let v = f.cast::<f32>();
                (v[0], v[1], v[2])
            })
            .collect();
        recording
            .log("/world/points", &rerun::Points3D::new(&tracked_pts))
            .unwrap();
        recording.log("/pose", &visualize::na_isometry3_to_rerun_transform3d(stereo_esimator.get_current_cam0_pose())).unwrap();
    }
    // println!("{:?}", stereo_esimator);
}
