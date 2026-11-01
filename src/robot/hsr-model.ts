import description from "./hsr-description.json" with { type: "json" };
import { Euler, Quaternion } from "three";

type Shape = { type: string; xyz: number[]; rpy: number[]; path?: string; vertices?: string; radius?: number[]; length?: number[]; size?: number[] };
type Link = { name: string; parent: string; joint: { name: string; type: string; xyz: number[]; rpy: number[];
  axis?: { xyz: string }; limit?: { lower: string; upper: string; velocity: string }; mimic?: { joint: string; multiplier?: string; offset?: string } };
  visuals: Shape[]; collisions: Shape[]; inertial: { mass: number; xyz: number[]; rpy: number[]; inertia: number[] } };
export const HSR_LINKS = description.links as Link[];
export const ARM_NAMES = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"];
export const ARM_LIMITS = [[0, 0.69], [-2.62, 0], [-2.09, 3.84], [-1.92, 1.22], [-1.92, 3.67]];
export const ARM_SPEEDS = [0.2, 1.2, 2, 1.5, 1.5];
export const BASE_HOME = [0.078, 1, 0] as const;
// Avoid the URDF's incompatible distal limit at motor=1.24 (-1.327 < -1.24).
export const HAND_OPEN = 1.153, HAND_CLOSED = -0.05;

export function rpyQuaternion(rpy: number[]): [number, number, number, number] {
  return new Quaternion().setFromEuler(new Euler(rpy[0], rpy[1], rpy[2], "ZYX")).toArray();
}
const pose = (value: { xyz: number[]; rpy: number[] }) => {
  const [x, y, z, w] = rpyQuaternion(value.rpy);
  return `pos="${value.xyz.join(" ")}" quat="${w} ${x} ${y} ${z}"`;
};
const meshName = (link: Link, i: number) => `${link.name}_collision_${i}`;
export const HSR_MESHES = HSR_LINKS.flatMap(link => link.collisions.flatMap((shape, i) => shape.type === "mesh"
  ? [`<mesh name="${meshName(link, i)}" vertex="${shape.vertices}"/>`] : [])).join("\n");

function body(link: Link): string {
  const { joint: j, inertial: inert } = link;
  const spring = j.name.includes("spring_proximal") ? ' stiffness="30" springref="0" damping="0.2"' : ' damping="0.5"';
  const moving = j.type !== "fixed";
  const shapes = link.collisions.map((shape, i) => {
    const name = link.name === "hand_l_distal_link" ? "left_pad_geom" : link.name === "hand_r_distal_link" ? "right_pad_geom" : meshName(link, i);
    const size = shape.type === "cylinder" ? `size="${shape.radius![0]} ${shape.length![0] / 2}"` :
      shape.type === "box" ? `size="${shape.size!.map(v => v / 2).join(" ")}"` : `mesh="${meshName(link, i)}"`;
    return `<geom name="${name}" type="${shape.type}" ${pose(shape)} ${size}/>`;
  }).join("\n");
  const children = HSR_LINKS.filter(child => child.parent === link.name).map(body).join("\n");
  // HSR hand-camera mount; display roll is aligned upright at the neutral hand pose.
  const site = link.name === "hand_palm_link" ? '<site name="tcp" pos="0 0 0.08" size="0.005" rgba="0 0 0 0"/><site name="hand_camera" pos="0.039 0 -0.0045" quat="0 0.7071067811865476 0.7071067811865476 0" size="0.003" rgba="0 0 0 0"/>' :
    link.name === "head_tilt_link" ? '<site name="head_camera" pos="-0.079796 0.022 0.215172" quat="0.5 0.5 -0.5 -0.5" size="0.003" rgba="0 0 0 0"/>' : "";
  return `<body name="${link.name}" ${pose(j)} gravcomp="0">
    ${moving ? `<joint name="${j.name}" type="${j.type === "prismatic" ? "slide" : "hinge"}" axis="${j.axis!.xyz}" range="${j.limit!.lower} ${j.limit!.upper}"${spring}/>` : ""}
    <inertial pos="${inert.xyz.join(" ")}" mass="${inert.mass}" fullinertia="${inert.inertia.join(" ")}"/>
    ${shapes}${site}${children}</body>`;
}
// One fixed coordinate conversion lets every articulated link retain its native ROS frame.
export const HSR_BODIES = `<body name="hsr_frame" quat="0.5 -0.5 0.5 0.5">${HSR_LINKS.filter(link => link.parent === "base_link").map(body).join("\n")}</body>`;
// The neck slides through the torso cover; finger hulls overlap their rotary mounts and same-side knuckles.
// Keep all other nonadjacent robot collisions, including arm/body and opposite fingers, active.
export const HSR_CONTACT_EXCLUSIONS = [
  ["base", "head_pan_link"],
  ...["l", "r"].flatMap(side => [
    ["wrist_roll_link", `hand_${side}_spring_proximal_link`],
    ["hand_palm_link", `hand_${side}_spring_proximal_link`],
    [`hand_${side}_spring_proximal_link`, `hand_${side}_distal_link`],
  ]),
].map(([first, second]) => `<exclude body1="${first}" body2="${second}"/>`).join("\n");
export const HSR_EQUALITIES = HSR_LINKS.flatMap(link => {
  const { joint: j } = link;
  return j.mimic ? [`<joint joint1="${j.name}" joint2="${j.mimic.joint}" polycoef="${j.mimic.offset ?? 0} ${j.mimic.multiplier ?? 1} 0 0 0" solref="0.004 1"/>`] : [];
}).join("\n");
export const HSR_ACTUATORS = [
  '<position name="arm_lift_joint" joint="arm_lift_joint" kp="1200" kv="80" forcerange="-100 100"/>',
  '<position name="arm_flex_joint" joint="arm_flex_joint" kp="220" kv="12" forcerange="-40 40"/>',
  ...["arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"].map(n => `<position name="${n}" joint="${n}" kp="220" kv="12" forcerange="-40 40"/>`),
  '<position name="hand_motor_joint" joint="hand_motor_joint" kp="20" kv="1" forcerange="-4 4"/>',
  ...["head_pan_joint", "head_tilt_joint"].map(n => `<position name="${n}" joint="${n}" kp="30" kv="4" forcerange="-6 6"/>`),
].join("\n");
