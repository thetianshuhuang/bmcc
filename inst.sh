sudo rm -rf build
sudo rm -rf dist
sudo rm -rf bmcc.egg-info
sudo pip3 uninstall bmcc -y
sudo python3 setup.py install --force

cd tests
python3 version.py
cd ..
