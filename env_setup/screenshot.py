''' 
Taking Screenshots with Appium
I'll be using Python and a sample iOS application from Apple's Developer Library
This tutorial assumes you understand how to run, launch, and interact with your application.
'''

from appium import webdriver
import os

desired_capabilities = {}
desired_capabilities['deviceName'] = 'Simulator'

capabilities = dict(
    platformName='Android',
    automationName='uiautomator2',
    deviceName='Android',
    newCommandTimeout="120000",
    adbExecTimeout="120000",
    noReset=True,
    uiautomator2ServerInstallTimeout="120000",
    uiautomator2ServerLaunchTimeout="120000",
    uiautomator2ServerReadTimeout="120000",
)
capabilities["udid"] = "emulator-5554"
from appium.options.android import UiAutomator2Options
options = UiAutomator2Options().load_capabilities(capabilities)
directory = '%s/' % os.getcwd()

appium_server_url = "http://0.0.0.0:4723"
driver = webdriver.Remote(appium_server_url, options=options)
file_name = 'screenshot.png'
driver.save_screenshot(directory + file_name)
print("screenshot saved to", directory + file_name)
