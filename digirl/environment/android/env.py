import os
import shutil
import subprocess, signal
import re
from time import sleep
import random
from .autoui_utils import autoui_prepare_prompt, AndroidAction, ActionType, ImageFeatureExtractor
import time
from digirl.misc import colorful_print

from appium import webdriver
from appium.options.android import UiAutomator2Options

import base64
from PIL import Image
from io import BytesIO
from termcolor import colored, cprint
import concurrent.futures
import numpy as np
import traceback

def escape_shell_text(text):
    # List of characters to escape
    chars_to_escape = ['\\','"', "'", '`', '$']
    
    # Escape the characters by adding a backslash before them
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    text = text.replace(" ", "%s")
    return text

def kill_all_emulators(adb_path, emulators=None):
    # Get the list of connected devices
    result = subprocess.run([adb_path, 'devices'], stdout=subprocess.PIPE)
    devices_output = result.stdout.decode('utf-8')
    
    # Find all emulator device names using a regular expression
    running_emulators = re.findall(r'emulator-\d+', devices_output)
    
    # Shut down each emulator found
    for emulator in emulators:
        if emulator not in running_emulators:
            continue
        subprocess.run([adb_path, '-s', emulator, 'emu', 'kill'])
        print(f'{emulator} has been shut down.')

    if not emulators:
        print("No running emulators found.")

def clone_avd(src_avd_name, tar_avd_name, android_avd_home):
    """
    Clone the source AVD to the target AVD.

    Parameters:
    - src_avd_name: The name of the source AVD folder.
    - tar_avd_name: The name of the target AVD folder.
    - android_avd_home: The path to the .android/avd directory.

    This function copies the source AVD folder and its .ini file to a new target AVD
    and updates the paths inside the .ini files accordingly.
    """

    # Paths for source and target AVD directories and .ini files
    src_avd_dir = os.path.join(android_avd_home, src_avd_name + '.avd')
    tar_avd_dir = os.path.join(android_avd_home, tar_avd_name + '.avd')
    src_ini_file = os.path.join(android_avd_home, src_avd_name + '.ini')
    tar_ini_file = os.path.join(android_avd_home, tar_avd_name + '.ini')

    # Copy the AVD folder
    colorful_print(f"Copying the AVD folder from {src_avd_dir} to {tar_avd_dir}", "green")
    if not os.path.exists(tar_avd_dir):
        shutil.copytree(src_avd_dir, tar_avd_dir)

    # Copy the .ini file and modify it for the new AVD
    with open(src_ini_file, 'r') as src_ini, open(tar_ini_file, 'w') as tar_ini:
        for line in src_ini:
            tar_ini.write(line.replace(src_avd_name, tar_avd_name))

    # Update paths inside the target AVD's .ini files
    for ini_name in ['config.ini', 'hardware-qemu.ini']:
        ini_path = os.path.join(tar_avd_dir, ini_name)
        if os.path.exists(ini_path):
            with open(ini_path, 'r') as file:
                lines = file.readlines()
            with open(ini_path, 'w') as file:
                for line in lines:
                    # Update paths and AVD name/ID
                    new_line = line.replace(src_avd_name, tar_avd_name)
                    file.write(new_line)

    # Update the snapshots' hardware.ini file if it exists
    snapshots_hw_ini = os.path.join(tar_avd_dir, 'snapshots', 'default_boot', 'hardware.ini')
    if os.path.exists(snapshots_hw_ini):
        with open(snapshots_hw_ini, 'r') as file:
            lines = file.readlines()
        with open(snapshots_hw_ini, 'w') as file:
            for line in lines:
                # Update AVD name/ID
                new_line = line.replace(src_avd_name, tar_avd_name)
                file.write(new_line)


class AndroidEmulator():
    def __init__(self, avd_name, max_steps, temp_path, evaluator, emulator_path="~/Android/Sdk/emulator/emulator", appium_server_url='http://localhost:4723', no_window=False, udid = None,
        feature_extractor = None, all_tasks = None, prepare_prompt = autoui_prepare_prompt, translate_action = None, save_images = False, task_id=None, task_split="train", sample_mode=None, record=False):
        """
        temp_path temporary path to store the images for evaluation
        """
        self.temp_path = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.avd_name = avd_name
        self.save_images = save_images
        self.image_id = str(time.time())
        port_number = udid.split("-")[-1]
        self.udid = udid
        cprint(colored(f"Starting the Emulator", "green"))
        command = f"""{self.emulator_path} -avd {self.avd_name} "-no-audio" "-skip-adb-auth" "-no-boot-anim" "-gpu" "auto" "-no-snapshot-save" -port {port_number}"""
        if no_window:
            command += " -no-window"
        print(f"executing command {command}")
        self.emulator_process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sleep(30)
        self.record = record
        if self.record:
            self.record_random_id = random.randint(0, 100000)
            try_record_command = f"""adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_1.mp4"""
            # redirect the output and error to the output of the main process
            import sys
            print(f"Trying to record the screen of {self.udid}")
            self.try_record_process = subprocess.Popen(try_record_command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
            sleep(20)
            self.try_record_process.terminate()
            try:
                self.try_record_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.try_record_process.kill()
                self.try_record_process.wait()
            sleep(5)
            print(f"Recording the screen of {self.udid}")
            do_record_command = f"""adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_1.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_2.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_3.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_4.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_5.mp4 &&
adb -s {self.udid} shell screenrecord --size 540x1140 --bit-rate 4M --time-limit=180 /sdcard/video_{self.image_id}_6.mp4"""
            self.record_process = subprocess.Popen(do_record_command, shell=True, stdout=subprocess.PIPE, preexec_fn=os.setsid) # should be good the second time
            sleep(5)

        capabilities = dict(
            platformName='Android',
            automationName='uiautomator2',
            deviceName='Android',
            newCommandTimeout="120000",
            adbExecTimeout="120000",
            uiautomator2ServerInstallTimeout="120000",
            uiautomator2ServerLaunchTimeout="120000",
            uiautomator2ServerReadTimeout="120000",
            noSign=True
        )
        if udid:
            capabilities["udid"] = udid
        self.options = UiAutomator2Options().load_capabilities(capabilities)
        self.appium_server_url = appium_server_url
        for i in range(3):
            try:
                self.driver = webdriver.Remote(self.appium_server_url, options=self.options)
                print("connected!")
                break
            except Exception as e:
                cprint(colored(f"Failed to connect to the appium server: {e}\n Retrying", "red"))
                if i == 3:
                    raise Exception("Failed to connect to the appium server")
                sleep(20)
        self.terminated = False
        self.max_steps = max_steps
        self.steps = 0
        self.feature_extractor = feature_extractor
        screen_size = self.driver.get_window_size()
        self.screen_size = (screen_size["width"], screen_size["height"])
        if sample_mode == "random":
            # randomly sample a task from the task set
            self.current_task = random.choice(all_tasks)
        elif sample_mode == "sequential":
            self.current_task = all_tasks[task_id]
        else:
            print("Invalid sample mode")
        self.prepare_prompt = prepare_prompt
        self.translate_action = translate_action
        self.history = []
        self.evaluator = evaluator
    
    def terminate(self):
        
        if self.record:
            # send sigterm to the record process
            os.killpg(os.getpgid(self.record_process.pid), signal.SIGINT)
            sleep(5)
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_1.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_2.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_3.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_4.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_5.mp4 {self.temp_path}")
            os.system(f"adb -s {self.udid} pull /sdcard/video_{self.image_id}_6.mp4 {self.temp_path}")
            print("it's okay if you see errros like failed to stat remote object '/sdcard/video_1718747809.256034_{i}.mp4' where i is larger than 1.")

        sleep(5)
        self.emulator_process.terminate()
        try:
            self.emulator_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.emulator_process.kill()
            self.emulator_process.wait()
        self.terminated = True
    
    def refresh_driver(self):
        self.driver.quit()
        self.driver = webdriver.Remote(self.appium_server_url, options=self.options)
    
    def count_white_pixels(self, img):
        # Convert the image to RGB format if it's not
        img = img.convert('RGB')
        # Convert image to numpy array
        data = np.array(img)
        # Count white pixels
        # Assuming 'white' is (255, 255, 255)
        white_count = np.sum(np.all(data > 240, axis=-1))
        return white_count > 2_300_000
    
    def get_obs(self):
        for _ in range(3):
            try:
                is_white = True
                for _ in range(5):
                    if not is_white:
                        break
                    sleep(5)
                    screenshot_str = self.driver.get_screenshot_as_base64()
                    imgdata = base64.b64decode(screenshot_str)
                    image =  Image.open(BytesIO(imgdata))
                    is_white = self.count_white_pixels(image)
                # print("Saving observation!")
                image.save(os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"))
                # Assuming 'image' is your PIL Image object in RGBA mode
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                if self.feature_extractor is not None:
                    image = self.feature_extractor.to_feat(image)
                # colorful_print(f"history: {self.history}", "green")
                # colorful_print(f"prompt: {self.prepare_prompt(self.current_task, self.history)}", "yellow")
                return {"prompt": self.prepare_prompt(self.current_task, self.history),
                        "image_feature": image,
                        "task": self.current_task,
                        "image_path": os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"),
                        "video_path": os.path.join(self.temp_path, f"video_{self.record_random_id}.mp4") if self.record else None
                }
            except Exception as e:
                print(f"Exception happened during screenshotting")
                print(e)
                print(traceback.format_exc())
                sleep(6)
                continue
    def step(self, raw_action: str):
        if self.terminated:
            return None
        try:
            # colorful_print(f"raw action: {raw_action}", "green")
            action = self.translate_action(raw_action)
            # colorful_print(f"translated action: {action}", "green")
        except Exception as e:
            print(e)
            print(f"Failed to translate action: {raw_action}, terminating the environment")
            action = AndroidAction(action_type=ActionType.TaskImpossible)
        self.history.append(action)
        self.steps += 1
        if self.steps > self.max_steps:
            action = AndroidAction(action_type=ActionType.TaskImpossible)
            cprint(colored(f"Terminate the Emulator: Max Steps Exceeded {self.max_steps}.", "red"))
        screenshot = None
        info = {}
        for i in range(2):
            try:
                if action.action_type == ActionType.DualPoint:
                    assert len(action.touch_point) == 2
                    assert len(action.lift_point) == 2
                    touch_x = action.touch_point[0] * self.screen_size[0]
                    touch_y = action.touch_point[1] * self.screen_size[1]
                    lift_x = action.lift_point[0] * self.screen_size[0]
                    lift_y = action.lift_point[1] * self.screen_size[1]
                    if (touch_x - lift_x)**2 + (touch_y - lift_y)**2 < 10:
                        self.driver.tap([(touch_x, touch_y)])
                    else:
                        self.driver.swipe(touch_x, touch_y, lift_x, lift_y)
                elif action.action_type == ActionType.Type:
                    # This doesn't work well because of active element
                    for i in range(2):
                        try:
                            sleep(4)
                            element = self.driver.switch_to.active_element
                            element.send_keys(action.typed_text)
                            break
                        except Exception as e:
                            cprint(f"The element is not loaded yet or agent did not click anything", "red")
                    
                elif action.action_type == ActionType.GoBack:
                    self.driver.back()
                elif action.action_type == ActionType.GoHome:
                    self.driver.press_keycode(3)
                elif action.action_type == ActionType.Enter:
                    self.driver.press_keycode(66)
                elif action.action_type == ActionType.TaskComplete:
                    self.terminated = True
                elif action.action_type == ActionType.TaskImpossible:
                    self.terminated = True
                elif action.action_type == ActionType.Idle:
                    pass
                else:
                    raise Exception(f"Unknown action type: {action.action_type}")
                action_success = True
                screenshot = self.get_obs()
                break
            except Exception as e:
                cprint(colored("an Exception occurred during environment interaction", "red"))
                print(e)
                cprint(colored("Retrying", "red"))
                sleep(10)
                if i == 1:
                    action_success = False
                    info["error"] = str(e)
                    self.driver.quit()
                    self.terminate()
                    return None
                continue
        r = 0
        if screenshot is not None and self.evaluator is not None:
            r = self.evaluator([os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}.png"), 
                                os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png")], self.current_task)
        info["action_success"] = action_success
        #terminate the environment if there is a success
        if r >= 1 or self.terminated:
            self.driver.quit()
            self.terminate()
        if self.terminated and not self.save_images:
            os.system(f"rm -rf {self.temp_path}/*")
        return screenshot, r, self.terminated


class BatchedAndroidEnv():
    """
    This class wraps around the android emulator and provides a more infrastructure for free-form GUI navigation
    This is a batched version for Android Env
    cache_avd is the avd to be used the avd is the initial one
    """
    def __init__(self, 
        avd_name, 
        cache_avd_names,
        udids,
        appium_base_port,
        android_avd_home: str = '/nfs/kun2/users/yifei/openended/.android/android_avd/avd',
        emulator_path: str = '~/Android/Sdk/emulator/emulator',
        adb_path: str = "~/Library/Android/sdk/platform-tools/adb",
        run_headless: bool = False,
        max_steps: int = 10,
        use_feature_extractor = False, 
        evaluators = None,
        prepare_prompt = autoui_prepare_prompt, 
        translate_action = None,
        device = "cuda:2",
        temp_path = "/nfs/kun2/users/yifei/openended/logs/images",
        save_images = False,
        all_tasks = None,
        task_split = "train",
        sample_mode = None,
        record = False):
        
        self.android_avd_home = os.path.expanduser(android_avd_home)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.adb_path = os.path.expanduser(adb_path)
        self.avd_name = avd_name
        self.save_images = save_images
        self.bsize = len(cache_avd_names)
        self.cache_avd_names = cache_avd_names
        self.run_headless = run_headless
        self.max_steps = max_steps
        self.emulator_group_offset = 0
        if use_feature_extractor:
            self.feature_extractor = ImageFeatureExtractor("cpu")
        else:
            self.feature_extractor = None
        self.device = device
        self.record = record
        self.all_tasks = all_tasks
        self.task_split = task_split
        self.prepare_prompt = prepare_prompt
        self.translate_action = translate_action
        self.temp_path = temp_path
        if evaluators is None:
            evaluators = [None for _ in range(self.bsize)]
        self.evaluators = evaluators
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.udids = udids
        self.base_port = appium_base_port
        self.appium_processes = []
        self.sample_mode = sample_mode

        # Start the appium servers
        for i in range(self.base_port, self.base_port+self.bsize):
            self.appium_processes.append(subprocess.Popen(f"appium --relaxed-security -p {i} > /dev/null", stdout=subprocess.DEVNULL, shell=True))
            print("starting appium server at port ", i)
        self.appium_server_urls = [f"http://localhost:{i}" for i in range(self.base_port, self.base_port+self.bsize)]
    
    def reset_appium(self):
        for p in self.appium_processes:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        os.system("pkill -f appium")
        self.base_port = self.base_port + self.bsize * 2
        self.appium_processes = []
        for i in range(self.base_port, self.base_port+self.bsize):
            self.appium_processes.append(subprocess.Popen(f"appium --relaxed-security -p {i} > /dev/null", stdout=subprocess.DEVNULL, shell=True))
        # sleep(10)
        self.appium_server_urls = [f"http://localhost:{i}" for i in range(self.base_port, self.base_port+self.bsize)]

    def reset(self):
        """
        Reset the emulator to a clean state
        """
        # If the emulator is already running, kill it,
        # Then delete the cache AVD
        kill_all_emulators(self.adb_path, emulators=self.udids)
        if hasattr(self, "emulator_process"):
            self.emulator_process.send_signal(signal.SIGINT)
            self.emulator_process.wait()
        self.emulators = []
        for cache_avd_name in self.cache_avd_names:
            # print(cache_avd_name)
            for _ in range(3):
                try:
                    cache_avd_path = os.path.join(self.android_avd_home, cache_avd_name + ".avd")
                    cache_avd_ini_path = os.path.join(self.android_avd_home, cache_avd_name + ".ini")
                    if os.path.exists(cache_avd_path):
                        shutil.rmtree(cache_avd_path, ignore_errors=True)
                    if os.path.exists(cache_avd_ini_path):
                        os.remove(cache_avd_ini_path)
                    sleep(2)
                    # Clone the source AVD and start the emulator
                    clone_avd(self.avd_name, cache_avd_name, self.android_avd_home)
                    break
                except OSError as e:
                    print(f"Failed to reset the emulator: {e}")
                    import traceback
                    print(traceback.format_exc())
                    sleep(20)

        # # use parallel version only when you've got nice CPUs, or it will error out
        # def reset_emulator(cache_avd_name, avd_name, android_avd_home):
        #     for _ in range(3):
        #         try:
        #             cache_avd_path = os.path.join(android_avd_home, cache_avd_name + ".avd")
        #             cache_avd_ini_path = os.path.join(android_avd_home, cache_avd_name + ".ini")
        #             if os.path.exists(cache_avd_path):
        #                 shutil.rmtree(cache_avd_path, ignore_errors=True)
        #             if os.path.exists(cache_avd_ini_path):
        #                 os.remove(cache_avd_ini_path)
        #             sleep(2)
        #             # Clone the source AVD and start the emulator
        #             clone_avd(avd_name, cache_avd_name, android_avd_home)
        #             break
        #         except OSError as e:
        #             print(f"Failed to reset the emulator: {e}")
        #             import traceback
        #             print(traceback.format_exc())
        #             sleep(20)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(reset_emulator, cache_avd_name, self.avd_name, self.android_avd_home) for cache_avd_name in self.cache_avd_names]
        #     for future in futures:
        #         future.result()

        def emulator_constructor(udid, appium_server_url, cache_avd_name, evaluator, task_id, task_split):
            return AndroidEmulator(avd_name=cache_avd_name, max_steps=self.max_steps, emulator_path=self.emulator_path, 
                appium_server_url=appium_server_url, 
                no_window=self.run_headless, 
                udid = udid,
                feature_extractor = self.feature_extractor,
                prepare_prompt = self.prepare_prompt,
                translate_action = self.translate_action,
                all_tasks = self.all_tasks,
                evaluator = evaluator,
                temp_path = os.path.join(self.temp_path, cache_avd_name),
                save_images = self.save_images,
                task_id=task_id,
                task_split=task_split,
                sample_mode=self.sample_mode,
                record=self.record)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator_constructor, udid, appium_server_url, cache_avd_name, evaluator, task_id, self.task_split)
                for udid, appium_server_url, cache_avd_name, evaluator, task_id in 
                zip(self.udids, self.appium_server_urls, self.cache_avd_names, self.evaluators, range(self.emulator_group_offset, self.emulator_group_offset+self.bsize))]
            self.emulators = [job.result() for job in jobs]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator.get_obs) for emulator in self.emulators]
            # for i, job in enumerate(jobs):
                # colorful_print(f"Getting observation from emulator {i}: {job.result()}", "green")
            return [job.result() for job in jobs]

    def step(self, actions):
        if not self.emulators:
            raise Exception("Please call reset() before calling step()")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator.step, action) 
                    for emulator, action in 
                    zip(self.emulators, actions)]
            results = [job.result() for job in jobs]
        return results
