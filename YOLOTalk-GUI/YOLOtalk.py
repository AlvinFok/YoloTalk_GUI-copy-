from flask import Flask, render_template, Response, request, send_file, render_template_string
from flask.helpers import get_root_path

from web_ultis import *
from config import Config
from web_ultis import Restart_YoloDevice
from libs.YOLO_SSIM import YoloDevice

import pathlib
import os
import json
import cv2
import time
import sys
sys.path.append("..")


app = Flask(__name__)

# Restart YoloDevice
actived_yolo = Restart_YoloDevice()


@app.route("/", methods=["GET", "POST"])
def home():

    all_fences_names = read_all_fences()

    if request.method == "POST":
        alias = request.form.get("area")
        URL = str(request.form.get("URL"))
        Addtime = str(time.asctime(time.localtime(time.time())))[4:-5]
        print(f"=== Home Page ===")
        print(f"alias   :\t{alias}")
        print(f"URL     :\t{URL}")
        print(f"Addtime :\t{Addtime}")

        postURL = GET_POST_URL("plotarea")

        if URL == "REPLOT":
            IMGpath, shape = replot(alias, URL, Addtime)
            return render_template(
                "plotarea.html",
                data=IMGpath,
                name=str(alias),
                shape=shape,
                postURL=postURL,
            )

        else:
            # time.sleep(1)   # 避免opencv反應慢
            print("Connecting URL...")
            fig = cv2.VideoCapture(URL)
            stat, I = fig.read()
            print(f"stat : {stat}")
            if stat == True:
                # Temporary information
                data = {"alias": "",
                        "viedo_url": "",
                        "add_time": "",
                        "fence": {}
                        }
                Fence_info = {
                    "vertex": "Full image",
                    "Group": "-",
                    "Data_File": "coco.data",
                    "Config_File": "yolov4.cfg",
                    "Weight_File": "yolov4.weights",
                    "Sensitivity": 0.5,
                    "Alarm_Level": "General",  # General, High
                    "Show_FPS": True,
                    "Show_Box": True,
                    "Schedule": {
                        "1": {"Start_time": "--:--", "End_time": "--:--"},
                    },
                }
                data["alias"] = alias
                data["viedo_url"] = URL
                data["add_time"] = Addtime
                data["fence"]["All"] = Fence_info

                filepath = "static/Json_Info/camera_info_" + \
                    data["alias"] + ".json"
                with open(filepath, "w", encoding="utf-8") as file:
                    json.dump(data, file, separators=(",\n", ":"), indent=4)

                IMGpath = "static/alias_pict/" + str(alias) + ".jpg"
                all_fences_names.append(str(alias))
                cv2.imwrite(IMGpath, I)

                # ======== FOR YOLO ========
                yolo1 = YoloDevice(
                    # darknet file, preset is coco.data(80 classes)
                    data_file="../cfg_person/coco.data",
                    config_file="../cfg_person/yolov4.cfg",  # darknet file, preset is yolov4
                    weights_file="../weights/yolov4.weights",  # darknet file, preset is yolov4
                    thresh=0.3,  # Yolo threshold, float, range[0, 1]
                    output_dir="./static/record/",  # Output dir for saving results
                    video_url=URL,  # Video url for detection
                    is_threading=True,  # Set False if the input is video file
                    vertex=None,  # vertex of fence, None -> get all image
                    alias=alias,    # Name the file and directory
                    display_message=False,  # Show the message (FPS)
                    obj_trace=True,  # Object tracking
                    save_img=True,  # Save image when Yolo detect
                    save_img_original=True,    # Save original image and results when Yolo detect
                    img_expire_day=1,   # Delete the img file if date over the `img_expire_day`
                    save_video=False,   # Save video including Yolo detect results
                    video_expire_day=1,  # Delete the video file if date over the `video_expire_day`
                    target_classes=["person"],  # Set None to detect all target
                    auto_restart=False,  # Restart the program when RTSP video disconnection
                    using_SSIM=False,    # Using SSIM to find the moving object
                    SSIM_debug=False,    # Draw the SSIM image moving object even Yolo have detected object
                )
                print(f"\n======== Activing YOLO , alias:{alias}========\n")
                yolo1.set_listener(on_data)
                yolo1.start()
                actived_yolo[alias] = yolo1
                # ======== FOR YOLO ========
                return render_template(
                    "plotarea.html",
                    data=IMGpath,
                    name=str(alias),
                    shape=I.shape,
                    postURL=postURL,
                )

            else:
                return render_template("home.html", navs=all_fences_names, alert=True)
    return render_template("home.html", navs=all_fences_names, alert=False)


@app.route("/plotarea", methods=["GET", "POST"])
def plotarea():

    postURL = GET_POST_URL("plotarea")

    if request.method == "POST":

        print("plotarea enter & POST")

        alias = request.form["alias"]
        FenceName = request.form["FenceName"]  # plot name
        vertex = request.form["vertex"]  # plot point
        oldName = request.form["oldName"]  # oldName

        IMGpath = "static/alias_pict/" + str(alias) + ".jpg"
        filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
        fig = cv2.imread(IMGpath)
        shape = fig.shape

        Fence_info = {
            "vertex": vertex,
            "Group": "-",
            "Data_File": "coco.data",
            "Config_File": "yolov4.cfg",
            "Weight_File": "yolov4.weights",
            "Sensitivity": 0.5,
            "Alarm_Level": "General",  # General, High
            "Show_FPS": True,
            "Show_Box": True,
            "Schedule": {
                "1": {"Start_time": "--:--", "End_time": "--:--"},
            },
        }

        if os.path.isfile(filepath):  # If file is exist
            with open(filepath, "r", encoding="utf-8") as f:
                Jdata = json.load(f)
                print(f"actived_yolo:{actived_yolo}")
            if vertex == "DELETE":
                Jdata["fence"].pop(FenceName)
                Fence_info["vertex"] = "Full image"
                Jdata["fence"]["All"] = Fence_info

            elif vertex == "Rename":
                Jdata["fence"][FenceName] = Jdata["fence"][oldName]  # copy
                Jdata["fence"].pop(oldName)  # del

            else:
                Jdata["fence"].pop("All")
                Jdata["fence"][FenceName] = Fence_info
                # ======== FOR YOLO ========
                old_vertex = vertex[1:-1]
                new_vertex = transform_vertex(old_vertex)
                data = {FenceName: new_vertex}
                actived_yolo[alias].vertex = data
                print(
                    f"\n======== Editing YOLO vertexs, alias:{alias} ,vertex:{actived_yolo[alias].vertex}========\n"
                )
                # ======== FOR YOLO ========
            print(f"NEW Jdata:{Jdata}")
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(Jdata, file, separators=(",\n", ":"), indent=4)

        else:
            print("\nFile doesn't exist !! \n")
            data["fence"][FenceName] = Fence_info
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",\n", ":"), indent=4)
            data["fence"] = {}

    return render_template("plotarea.html", data=IMGpath, name=str(alias), shape=shape, postURL=postURL)


@app.route("/management", methods=["GET", "POST"])
def management():

    all_fences_names = read_all_fences()
    postURL = GET_POST_URL("management")

    # nav Replot
    if request.method == "POST":

        alias = request.form.get("area")
        URL = request.form.get("URL")
        Addtime = str(time.asctime(time.localtime(time.time())))[4:-5]

        if URL == "REPLOT":
            IMGpath, shape = replot(alias, URL, Addtime)
            postURL = GET_POST_URL("plotarea")
            return render_template(
                "plotarea.html",
                data=IMGpath,
                name=str(alias),
                shape=shape,
                postURL=postURL,
            )

        if URL == "Edit":

            Alias = request.form.get("Alias")
            FenceName = request.form.get("FenceName")
            Group = request.form.get("Group")
            # Data = request.form.get("Model")
            Config = request.form.get("Config")
            Model = request.form.get("Model")
            Sensitivity = request.form.get("Sensitivity")
            Alarm_Level = request.form.get("Alarm_Level")
            Show_FPS = request.form.get("Show_FPS")
            Show_Box = request.form.get("Show_Box")

            filepath = "static/Json_Info/camera_info_" + str(Alias) + ".json"
            with open(filepath, "r", encoding="utf-8") as f:
                Jdata = json.load(f)

            # Change the json directly
            Jdata["fence"][FenceName]["Group"] = Group
            Jdata["fence"][FenceName]["Alarm_Level"] = Alarm_Level
            Jdata["fence"][FenceName]["Show_FPS"] = Show_FPS
            Jdata["fence"][FenceName]["Show_Box"] = Show_Box

            # Adjust the written parameters
            if Model in ['yolov4', 'yolov4-tiny', 'yolov7', 'yolov7-tiny']:
                data = Model + '.data'
                Config = Model + '.cfg'
                model = Model + '.weights'
            else:
                data = Model.replace('.weights', '.data')
                Config = Model.replace('.weights', '.cfg')
                model = Model
            old_data = Jdata["fence"][FenceName]["Data_File"]
            old_Config = Jdata["fence"][FenceName]["Config_File"]
            old_model = Jdata["fence"][FenceName]["Weight_File"]
            Jdata["fence"][FenceName]["Data_File"] = data
            Jdata["fence"][FenceName]["Config_File"] = Config
            Jdata["fence"][FenceName]["Weight_File"] = model
            # Judge sensitivity is changed or not
            old_Sensitivity = Jdata["fence"][FenceName]["Sensitivity"]
            Jdata["fence"][FenceName]["Sensitivity"] = Sensitivity
            if old_Sensitivity != Sensitivity:
                print(
                    f"actived_yolo[alias].thresh : {actived_yolo[Alias].thresh}")
                actived_yolo[Alias].thresh = float(Sensitivity)
            """
            更改model
            """
            if old_data != data or old_Config != Config or old_model != model:
                
                actived_yolo[Alias].data_file="../cfg_person/coco.data"
                actived_yolo[Alias].config_file="../cfg_person/" + Config
                actived_yolo[Alias].weights_file="../weights/" + model
                actived_yolo[Alias].init_Yolo()

            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(Jdata, file, separators=(",\n", ":"), indent=4)

            return render_template("management.html", navs=all_fences_names, postURL=postURL)

        if URL == "Delete":

            Alias = request.form.get("Alias")
            FenceName = request.form.get("FenceName")
            print("Delete Alias : ", Alias)

            filepath = "static/Json_Info/camera_info_" + str(Alias) + ".json"
            with open(filepath, "r", encoding="utf-8") as f:
                Jdata = json.load(f)

            Jdata["fence"].pop(FenceName)
            print("\n Jdata :", Jdata, "\n")

            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(Jdata, file, separators=(",\n", ":"), indent=4)

    # management items
    items = []
    filepath_list = os.listdir("./static/Json_Info/")
    for path in filepath_list:
        if ".json" in path:
            path = "./static/Json_Info/" + path
            with open(path, "r", encoding="utf-8") as f:
                file = json.load(f)
                items.append(file)
    # weights files
    weights = []
    weightspath_list = os.listdir("../weights")
    for path in weightspath_list:
        if ".weights" in path:
            weights.append(path.replace(".weights", ""))
    weights.sort()
    return render_template("management.html", navs=all_fences_names, items=items, weights=weights, postURL=postURL)


@app.route("/streaming", methods=["GET", "POST"])
def streaming():

    all_fences_names = read_all_fences()

    if request.method == "POST":

        alias = request.form.get("area")
        URL = request.form.get("URL")
        Addtime = str(time.asctime(time.localtime(time.time())))[4:-5]

        if URL == "REPLOT":

            IMGpath, shape = replot(alias, URL, Addtime)
            postURL = GET_POST_URL("plotarea")
            return render_template("plotarea.html", data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

    alias_list = os.listdir(r"static/alias_pict")
    for alias in alias_list:
        if '.gitignore' in alias:
            alias_list.remove(alias)
    alias_list.sort()

    return render_template(
        "streaming.html",
        navs=all_fences_names,
        alias_list=alias_list,
        length=len(alias_list),
    )


@app.route("/schedule", methods=["GET", "POST"])
def schedule():

    all_fences_names = read_all_fences()
    postURL = GET_POST_URL("schedule")

    if request.method == "POST":

        alias = request.form.get("area")
        URL = request.form.get("URL")
        Addtime = str(time.asctime(time.localtime(time.time())))[4:-5]

        if URL == "REPLOT":
            IMGpath, shape = replot(alias, URL, Addtime)
            postURL = GET_POST_URL("plotarea")
            return render_template(
                "plotarea.html",
                data=IMGpath,
                name=str(alias),
                shape=shape,
                postURL=postURL,
            )

        if URL == "Edit_time":
            alias = str(request.form.get("alias"))
            FenceName = request.form.get("FenceName")
            # Group       = request.form.get('Group')
            Order = str(request.form.get("Order"))
            Start_time = request.form.get("start_time")
            End_time = request.form.get("end_time")
            new_schedule = {"Start_time": Start_time, "End_time": End_time}

            filepath = f"static/Json_Info/camera_info_{alias}.json"
            with open(filepath, "r", encoding="utf-8") as f:
                Jdata = json.load(f)
            Schedule_keys = list(Jdata["fence"][FenceName]["Schedule"].keys())

            if Order in Schedule_keys:
                # Jdata['fence'][FenceName]['Group'] = Group
                Jdata["fence"][FenceName]["Schedule"][Order]["Start_time"] = Start_time
                Jdata["fence"][FenceName]["Schedule"][Order]["End_time"] = End_time
            else:
                # data['fence'][FenceName]['Group'] = Group
                Jdata["fence"][FenceName]["Schedule"][Order] = new_schedule
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(Jdata, file, separators=(",\n", ":"), indent=4)

        if URL == "Delete_Schedule":
            alias = request.form.get("alias")
            FenceName = request.form.get("FenceName")
            Order = request.form.get("Order")
            filepath = f"static/Json_Info/camera_info_{alias}.json"
            with open(filepath, "r", encoding="utf-8") as f:
                Jdata = json.load(f)

            Jdata["fence"][FenceName]["Schedule"][Order]["Start_time"] = "--:--"
            Jdata["fence"][FenceName]["Schedule"][Order]["End_time"] = "--:--"

            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(Jdata, file, separators=(",\n", ":"), indent=4)
    items = []
    filepath_list = os.listdir("./static/Json_Info/")
    for path in filepath_list:
        if ".json" in path:
            path = "./static/Json_Info/" + path
            with open(path, "r", encoding="utf-8") as f:
                file = json.load(f)
                items.append(file)

    return render_template(
        "schedule.html", navs=all_fences_names, items=items, postURL=postURL
    )


@app.route("/training_page", methods=["GET", "POST"])
def training():
    all_fences_names = read_all_fences()
    postURL = GET_POST_URL("training_page")

    if request.method == "POST":
        alias = request.form.get("area")
        URL = str(request.form.get("URL"))
        Addtime = str(time.asctime(time.localtime(time.time())))[4:-5]

        if URL == "REPLOT":
            IMGpath, shape = replot(alias, URL, Addtime)
            return render_template("plotarea.html", data=IMGpath, name=str(alias), shape=shape, postURL=postURL)
        
    else:
        alias = request.form.get("area")
        aliases = os.listdir('static/record')
        folders = os.listdir(f'static/record/{aliases[0]}/img_original')
        folders.sort()

        tmp_folders = []
        for folder in folders:
            dirs_day = os.listdir(f'static/record/{aliases[0]}/img_original/{folder}')
            for hour in dirs_day:
                tmp_folders.append(f'static/record/{aliases[0]}/img_original/{folder}/{hour}')
        tmp_folders.sort()

        All_img_files = []
        All_txt_files = []
        for tmp_folder in tmp_folders:
            files = os.listdir(tmp_folder)
            files.sort()

            tmp_img = []
            tmp_txt = []
            for file in files:
                if '.jpg' in file:
                    tmp_img.append(file)
                elif '.txt' in file:
                    with open(f'{tmp_folder}/{file}') as f:
                        lines_txt = []
                        for line in f.readlines():
                            # print(f'path = {tmp_folder}/{file}\n')
                            # print(f'line = {line.split()}\n')
                            lines_txt.append(line.split())
                        tmp_txt.append(lines_txt)

            All_img_files.append(tmp_img)
            All_txt_files.append(tmp_txt)

        IMGpath = tmp_folders[0] + '/' + All_img_files[0][0]
        first_fig = cv2.imread(IMGpath)
        shape = first_fig.shape

    return render_template("training.html", data=IMGpath, aliases=aliases, folders=tmp_folders, img_files=All_img_files, txt_files=All_txt_files,shape=shape, navs=all_fences_names, postURL=postURL)


@app.route("/", defaults={"req_path": ""})
@app.route("/<path:req_path>")
def dir_listing(req_path):
    all_fences_names = read_all_fences()

    if req_path == "favicon.ico":
        return "Error"
    elif "logo.png" in req_path:
        abs_path = "static/logo.png"
        return send_file(abs_path)
    else:
        BASE_DIR = "./static"  # The static path under the Flask

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)
    if not os.path.exists(abs_path):  # Return 404 if path doesn't exist
        print("Error")
    if os.path.isfile(abs_path):  # Check if path is a file and serve
        return send_file(abs_path)
    files = os.listdir(abs_path)  # Show directory contents
    files.sort()
    return render_template("files.html", files=files, navs=all_fences_names)


@app.route("/video/<order>", methods=["GET", "POST"])
def video_feed(order):
    alias_list = os.listdir(r"static/alias_pict")
    for alias in alias_list:
        if '.gitignore' in alias:
            alias_list.remove(alias)
    alias_list.sort()
    if len(actived_yolo) > int(order):
        print(f"actived_yolo len = {len(actived_yolo)}, order = {order}")
        alias = alias_list[int(order)].replace(".jpg", "")
        return Response(
            gen_frames(actived_yolo[alias]),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return "Error"
    
def train_model(darknet_path, data_path, cfg_path, weights_path, output_path, mode="train", thresh=0.5):
    """
    darknet_path: darknet 資料夾與current working dir 的相對路徑
    data_path: .data file和darknet資料夾的相對路徑
    cfg_path: .cfg file和darknet資料夾的相對路徑
    weights_path: .weights file和darknet資料夾的相對路徑
    mode: "train" 表示使用 ./darkent detector train 指令
          "valid" 表示使用 ./darkent detector map 指令
    thresh: ./darkent detector map 指令的 threshold, 預設為0.5
    -------------------------------------------------------------
    return 產生的 ./darknet process 的 ID
    """
    current_working_dir = os.getcwd()
    os.chdir(darknet_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(data_path) or not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        if not os.path.exists(data_path):
            print(f"file: {data_path} does not exists.")
        if not os.path.exists(cfg_path):
            print(f"file: {cfg_path} does not exists.")
        if not os.path.exists(weights_path):
            print(f"file: {weights_path} does not exists.")
        os.chdir(current_working_dir)
    else:
        os.chdir(current_working_dir)
        if mode == "train":
            command = [f"./darknet detector train {data_path} {cfg_path} {weights_path} -map -dont_show > {output_path}"]
        elif mode == "valid":
            command = [f"./darknet detector map {data_path} {cfg_path} {weights_path} -thresh {thresh} > {output_path}"]
        else:
            command = [""]
            print("Invalid mode. Please use \"train\" or \"valid\" string.")
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True, cwd= darknet_path)
        pid = subprocess.check_output(["pidof -s ./darknet"], shell=True)

    return int(pid.decode("utf-8"))

if __name__ == "__main__":
    app.run(debug=Config["DEBUG"], use_reloader=Config["use_reloader"], host=Config["host"], port=Config["port"],)