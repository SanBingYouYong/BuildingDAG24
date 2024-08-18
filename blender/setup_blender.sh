# apt install wget
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
# apt-get install xz-utils
tar -xf blender-3.2.2-linux-x64.tar.xz

# Archive Blender's own Python directory
tar -czf blender_python_backup.tar.gz -C blender-3.2.2-linux-x64/3.2 python/

# Remove the Python directory after archiving
rm -rf blender-3.2.2-linux-x64/3.2/python/

# Remove the download
rm blender-3.2.2-linux-x64.tar.xz

# instal dependencies
# apt install libxrender-dev
# apt install libgl-dev
# apt --fix-broken install
# apt install libxfixes3
# apt install libxxf86vm-dev
# apt-get install libxi6
