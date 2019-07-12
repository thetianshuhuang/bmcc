# Install
pip3 install .

# Check version
echo "System time:"
now=$(date)
echo "$now"
echo "Build time: should match"
cd examples
python3 -c "import bmcc; print(bmcc.CONFIG['BUILD_DATETIME'])"
cd ..
echo ""
