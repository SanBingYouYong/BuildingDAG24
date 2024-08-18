# apt install wget
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz

# Archive Blender's own Python directory
tar -czf blender_python_backup.tar.gz -C blender-3.2.2-linux-x64/3.2 python/

# Remove the Python directory after archiving
rm -rf blender-3.2.2-linux-x64/3.2/python/
