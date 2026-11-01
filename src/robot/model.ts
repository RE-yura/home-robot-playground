import { TABLE_HEIGHT, TABLE_THICKNESS } from "../simulation/table.ts";
import { ENVIRONMENT_BODIES, ENVIRONMENT_MESHES } from "../simulation/environment.ts";
import { HOUSEHOLD_BODIES, DRINK_BODIES, MUG_FLOOR_CONTACTS } from "../simulation/household.ts";
import { BASE_HOME, HSR_MESHES, HSR_BODIES, HSR_EQUALITIES, HSR_ACTUATORS, HSR_CONTACT_EXCLUSIONS } from "./hsr-model.ts";

// Existing Playground dimensions, Y up. All moving links are force-driven joints.
export const SCENE = [
  '<mujoco model="home-robot-playground"><compiler angle="radian"/>',
  '<option timestep="0.002" gravity="0 -9.81 0" integrator="implicitfast" cone="elliptic" iterations="30"/>',
  '<default><joint damping="0.3" armature="0.02"/>',
  '<geom friction="1.2 0.02 0.001" condim="4" solref="0.004 1" solimp="0.99 0.999 0.001" rgba="0.93 0.95 0.92 1"/>',
  '<default class="robot"><geom contype="1" conaffinity="3"/></default></default>',
  '<asset>' + HSR_MESHES + ENVIRONMENT_MESHES + '</asset><worldbody>',
  '<geom name="floor" type="plane" size="5 5 0.1" quat="0.707107 -0.707107 0 0" rgba="0.87 0.87 0.84 1" contype="2" conaffinity="3"/>',
  `<geom name="table" type="box" pos="0 ${TABLE_HEIGHT - TABLE_THICKNESS / 2} 0" size="0.8 ${TABLE_THICKNESS / 2} 0.5" rgba="0.55 0.34 0.19 1" contype="2" conaffinity="3"/>`,
  ...[-0.66, 0.66].flatMap((x) => [-0.36, 0.36].map((z) =>
    `<geom type="box" pos="${x} ${(TABLE_HEIGHT - TABLE_THICKNESS) / 2} ${z}" size="0.035 ${(TABLE_HEIGHT - TABLE_THICKNESS) / 2} 0.035" rgba="0.3 0.35 0.33 1" contype="2" conaffinity="3"/>`)),
  `<body name="cube" pos="-0.55 ${TABLE_HEIGHT + 0.34} -0.26"><freejoint/><geom name="cube_geom" type="box" size="0.065 0.065 0.065" mass="1" rgba="0.78 0.44 0.3 1" contype="2" conaffinity="3"/></body>`,
  `<body name="ball" pos="-0.23 ${TABLE_HEIGHT + 0.34} 0.2"><freejoint/><geom type="sphere" size="0.0335" mass="0.058" contype="2" conaffinity="3"/></body>`,
  '<body name="baseball" pos="0.65 0.3 1.6"><freejoint/><geom type="sphere" size="0.0365" mass="0.145" contype="2" conaffinity="3"/></body>',
  `<body name="can" pos="-0.53 ${TABLE_HEIGHT + 0.36} 0.1"><freejoint/><geom type="cylinder" size="0.05 0.08" quat="0.707107 0.707107 0 0" mass="0.8" rgba="0.84 0.67 0.29 1" contype="2" conaffinity="3"/></body>`,
  ...HOUSEHOLD_BODIES,
  ...DRINK_BODIES,
  ...ENVIRONMENT_BODIES,
  '<body name="base" pos="' + BASE_HOME[0] + ' 0 ' + BASE_HOME[1] + '" childclass="robot">',
  '<joint name="base_x" type="slide" axis="1 0 0" damping="5"/><joint name="base_z" type="slide" axis="0 0 1" damping="5"/>',
  '<joint name="base_yaw" type="hinge" axis="0 1 0" damping="3"/>',
  '<joint name="base_y" type="slide" axis="0 1 0" damping="0.2"/><joint name="base_pitch" type="hinge" axis="1 0 0" damping="0.3"/><joint name="base_roll" type="hinge" axis="0 0 1" damping="0.3"/>',
  '<geom type="cylinder" pos="0 0.126 0" size="0.229 0.103" quat="0.707107 0.707107 0 0" mass="15" rgba="0.19 0.23 0.22 1"/>',
  // Rounded supports approximate the omni base's free rolling while carrying its real normal loads.
  ...[-0.145, 0.145].flatMap((x, i) => [-0.145, 0.145].map((z, j) =>
    `<geom name="base_support_${i}_${j}" type="sphere" pos="${x} 0.018 ${z}" size="0.018" mass="0" priority="1" condim="3" friction="0.025 0.002 0.0002" rgba="0.19 0.23 0.22 1"/>`)),
  '<geom type="box" pos="0 0.431 0.073" size="0.181 0.289 0.109" mass="5"/>',
  '<geom type="box" pos="0.00825 0.323 -0.008" size="0.05525 0.19 0.025" mass="1" rgba="0.44 0.5 0.49 1"/>',
  '<geom type="cylinder" pos="0 0.545 0" size="0.045 0.035" quat="0.707107 0.707107 0 0" mass="1"/>',
  HSR_BODIES,
  '</body></worldbody><contact>' + HSR_CONTACT_EXCLUSIONS + MUG_FLOOR_CONTACTS + '</contact><equality>' + HSR_EQUALITIES + '</equality><actuator>',
  ...['base_x', 'base_z'].map((n) => '<position name="' + n + '" joint="' + n + '" kp="800" kv="180" forcerange="-150 150"/>'),
  '<position name="base_yaw" joint="base_yaw" kp="150" kv="35" forcerange="-60 60"/>',
  HSR_ACTUATORS,
  '</actuator></mujoco>',
].join('\n');
