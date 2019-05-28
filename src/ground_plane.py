print('<mujoco model="ground_plane">')
print('  <worldbody>')
print('    <geom conaffinity="1" condim="3" name="floor" friction="0.8 0.1 0.1" pos="0 0 0" type="plane"/>')

STAIRS_PER_SET = 6
STAIRS_SET = 5

top_size=0.5

color_delta = 1 / STAIRS_PER_SET
for i in range(STAIRS_SET):
    height = 0.01*(i+1)
    pos_offset = 8*(i+1)

    for j in range(i+1):
        print('<geom type="cylinder" conaffinity="1" condim="3" friction="0.8 0.8 0.8" rgba="'+str(j*color_delta) +'   0 ' +str( (STAIRS_PER_SET - j) * color_delta) +'  0" pos="' + str(pos_offset) + '  0 ' + str(height*(2*j+1) )+'" size="'+ str((STAIRS_PER_SET - j)*0.33 + top_size)+' '+ str(height)+'" name="top_step-'+str(i)+'-'+str(j)+'"/>')
        #print('<geom type="cylinder" conaffinity="1" condim="3" friction="0.8 0.1 0.1" rgba="'+str(j*color_delta) +'   1 ' +str( (STAIRS_PER_SET - j) * color_delta) +'  0" pos="' + str(pos_offset) + '  0 ' + str(height*(2*j+1) )+'" size="'+ str((STAIRS_PER_SET - j)*0.33 + top_size - 0.001)+' '+ str(height + 0.001)+'" name="top_step-'+str(i)+'-'+str(j)+'"/>')

print('  </worldbody>')
print('</mujoco>')
