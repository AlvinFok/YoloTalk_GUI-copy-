from libs.YOLO_SSIM import YoloDevice
from libs.tracker import Tracker


def on_data(save_path_img: str, group: str, alias: str, detect: str):
    tracker.update(detect, yolo1.frame)

video = "./0325__12__0.mp4"
alias = "test"
tracker = Tracker(
            enable_people_counting=True,
            track_buffer=30,
            tracker_mode="BYTE",
            in_area=[[0, 1100],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1100, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1100] ],#count in vertex
            out_area=[[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1609, 578], [1980, 728], [1980, 1080]],#count out vertex
            )
yolo1 = YoloDevice(
            # darknet file, preset is coco.data(80 classes)
            data_file="./cfg_person/coco.data",
            config_file="./cfg_person/yolov4.cfg",  # darknet file, preset is yolov4
            weights_file="./weights/yolov4.weights",  # darknet file, preset is yolov4
            thresh=0.3,  # Yolo threshold, float, range[0, 1]
            output_dir="./YOLOTalk-GUI/static/record/",  # Output dir for saving results
            video_url=video,  # Video url for detection
            is_threading=False,  # Set False if the input is video file
            vertex=None,  # vertex of fence, None -> get all image
            alias=alias,    # Name the file and directory
            display_message=False,  # Show the message (FPS)
            save_img=False,  # Save image when Yolo detect
            save_img_original=True,    # Save original image and results when Yolo detect
            img_expire_day=1,   # Delete the img file if date over the `img_expire_day`
            save_video=False,   # Save video including Yolo detect results
            video_expire_day=1,  # Delete the video file if date over the `video_expire_day`
            target_classes=["person"],  # Set None to detect all target
            auto_restart=False,  # Restart the program when RTSP video disconnection
            using_SSIM=False,    # Using SSIM to find the moving object
            SSIM_debug=False,    # Draw the SSIM image moving object even Yolo have detected object
            
        )
print(f"\n======== Activating YOLO , alias:{alias}========\n")
yolo1.set_listener(on_data)
yolo1.start()