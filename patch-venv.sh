rm VENV_DIR/lib/python3.5/site-packages/roboschool/mujoco_assets/ground_plane.xml
ln -s ../../../../../../src/ground_plane.xml VENV_DIR/lib/python3.5/site-packages/roboschool/mujoco_assets/ground_plane.xml

rm VENV_DIR/lib/python3.5/site-packages/roboschool/gym_forward_walker.py
ln -s ../../../../../src/gym_forward_walker.py VENV_DIR/lib/python3.5/site-packages/roboschool/gym_forward_walker.py
