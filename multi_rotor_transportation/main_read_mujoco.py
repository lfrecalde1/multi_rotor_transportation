import mujoco
import mujoco.viewer
import numpy as np

def main():
    model = mujoco.MjModel.from_xml_path("/home/ros2_ws/src/multi_rotor_transportation/model/simple_drone_payload_mujoco.xml")
    data = mujoco.MjData(model)

    geometric_names = [] 
    body_names = [] 
    body_id = []
    for i in range(model.ngeom):
        geometric_names.append(model.geom(i).name)

    for i in range(model.nbody):
        body_names.append(model.body(i).name)
    
    for i in range(model.nbody):
        body_id.append(model.body(i).id)
        print(model.body(i).name)
        print(model.body_inertia[i])
        print(model.body_mass[i])


    return None

if __name__ == '__main__':
    main()