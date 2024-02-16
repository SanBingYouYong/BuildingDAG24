import bpy

preferences = bpy.context.preferences
cycles_preferences = preferences.addons["cycles"].preferences
cycles_preferences.refresh_devices()
devices = cycles_preferences.devices
print(devices.items())
