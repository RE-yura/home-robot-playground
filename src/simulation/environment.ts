import type { Vec3 } from "./types.ts";
import { TABLE_HEIGHT } from "./table.ts";
import diningChair from "./dining-chair-collision.json" with { type: "json" };
import basket from "./basket-shapes.json" with { type: "json" };
import tvPlant from "./tv-plant-collision.json" with { type: "json" };

type Box = [number, number, number, number, number, number]; // local center + half extents, metres
export const FRIDGE_DOOR = { hinge: [0.41, 0.66, 0.426] as Vec3, handle: [-0.7, -0.015, 0.089] as Vec3,
  massKg: 6, maxAngle: 100 * Math.PI / 180 };
export const KITCHEN_DOOR = { hinge: [-0.68, 0.45, 0.37] as Vec3, handle: [0.24, 0.33, 0.024] as Vec3,
  massKg: 2, maxAngle: 105 * Math.PI / 180 };
export const OVEN_DOOR = { hinge: [0.25, 0.185, 0.36] as Vec3, handle: [0, 0.48, 0.059] as Vec3,
  massKg: 2.5, maxAngle: Math.PI / 2 };
// Wall centres and inward-facing local +Z are shared by rendering and contact.
export const ROOM_WALLS: { id: string; position: Vec3; rotation: number; width: number }[] = [
  { id: "back_wall", position: [0, 1.35, -3.175], rotation: 0, width: 6.6 },
  { id: "left_wall", position: [-3.325, 1.35, -0.2], rotation: Math.PI / 2, width: 6.0 },
  { id: "front_wall", position: [0, 1.35, 2.775], rotation: Math.PI, width: 6.6 },
  { id: "right_wall", position: [3.325, 1.35, -0.2], rotation: -Math.PI / 2, width: 6.0 },
];
// One placement for rendering and collision. Movable bodies retain their open contact shapes.
export const FURNITURE: { id: string; asset: string; position: Vec3; rotation?: number; size?: Vec3; origin?: "native"; boxes: Box[]; massKg?: number; friction?: number }[] = [
  { id: "basket", asset: "generated/bin.gltf", position: [0.48, TABLE_HEIGHT + 0.025, -0.2], origin: "native", massKg: 1.5, friction: 0.65,
    boxes: basket as Box[] },
  { id: "cabinet", asset: "models/modern_wooden_cabinet/model.gltf", position: [-0.9, 0, -2.73], size: [2, 0.77, 0.58],
    boxes: [[0, 0.425, 0, 1, 0.345, 0.29], ...[-0.89, 0.89].flatMap(x => [-0.23, 0.23].map(z => [x, 0.04, z, 0.035, 0.04, 0.035] as Box))] },
  { id: "kitchen", asset: "generated/kitchen.gltf", position: [1.13, 0, -2.72], origin: "native",
    boxes: [[-0.71, 0.45, 0, 0.015, 0.44, 0.34], [0.71, 0.45, 0, 0.015, 0.44, 0.34],
      [0, 0.45, -0.325, 0.71, 0.44, 0.015], [0, 0.035, 0, 0.71, 0.025, 0.34],
      [-0.44, 0.45, 0, 0.24, 0.008, 0.32], [-0.155, 0.45, 0, 0.025, 0.44, 0.34],
      [0.575, 0.45, 0, 0.025, 0.44, 0.34], [0.25, 0.145, 0, 0.30, 0.025, 0.34],
      [0.25, 0.78, 0.375, 0.3, 0.05, 0.025], [0.25, 0.81, 0, 0.3, 0.02, 0.34],
      [0, 0.92, 0, 0.765, 0.0325, 0.375], [0.25, 0.962, 0, 0.295, 0.008, 0.255],
      [0, 1.18, -0.347, 0.75, 0.22, 0.009], [0.25, 1.92, -0.13, 0.325, 0.04, 0.225], [0.25, 2.28, -0.28, 0.125, 0.34, 0.09]] },
  { id: "fridge", asset: "generated/fridge.gltf", position: [2.57, 0, -2.64], origin: "native",
    boxes: [[-0.408, 0.94, 0, 0.027, 0.93, 0.414], [0.408, 0.94, 0, 0.027, 0.93, 0.414],
      [0, 0.94, -0.39, 0.381, 0.93, 0.03], [0, 0.067, 0, 0.381, 0.057, 0.414],
      [0, 1.848, 0, 0.381, 0.022, 0.414], [0, 1.245, 0, 0.381, 0.02, 0.414],
      ...[0.40, 0.77, 1.12].map(y => [0, y, -0.015, 0.372, 0.006, 0.35] as Box),
      [0, 1.54, 0.448, 0.41, 0.269, 0.0225], [-0.29, 1.525, 0.515, 0.012, 0.115, 0.012]] },
  { id: "shelf", asset: "generated/shelf.gltf", position: [-2.68, 0, -2.70], origin: "native",
    boxes: [[-0.435, 1, 0, 0.0175, 1, 0.21], [0.435, 1, 0, 0.0175, 1, 0.21], [0, 1, -0.20, 0.435, 1, 0.01],
      ...[0.04, 0.53, 1.02, 1.51, 1.98].map(y => [0, y, 0, 0.435, 0.02, 0.21] as Box), [-0.10, 1.2, 0, 0.25, 0.16, 0.14]] },
  { id: "sofa", asset: "generated/sofa.gltf", position: [-2.71, 0, -0.63], rotation: Math.PI / 2, origin: "native",
    boxes: [[0, 0.36, 0, 0.82, 0.17, 0.40], [0, 0.73, -0.31, 0.82, 0.26, 0.1],
      [-0.875, 0.515, 0, 0.075, 0.285, 0.41], [0.875, 0.515, 0, 0.075, 0.285, 0.41],
      ...[-0.78, 0.78].flatMap(x => [-0.29, 0.29].map(z => [x, 0.1, z, 0.04, 0.1, 0.04] as Box))] },
  { id: "coffee_table", asset: "models/modern_coffee_table_01/model.gltf", position: [-1.76, 0, -0.60], size: [0.42, 0.39, 0.85],
    boxes: [[0, 0.37, 0, 0.21, 0.02, 0.425], ...[-0.16, 0.16].flatMap(x => [-0.36, 0.36].map(z => [x, 0.175, z, 0.02, 0.175, 0.02] as Box))] },
  { id: "armchair", asset: "models/modern_arm_chair_01/model.gltf", position: [-2.70, 0, 1.27], rotation: Math.PI / 2, size: [0.72, 0.9, 0.78], massKg: 12,
    boxes: [[0, 0.45, 0, 0.3, 0.07, 0.35], [0, 0.70, -0.3, 0.36, 0.2, 0.09],
      [-0.32, 0.59, 0, 0.04, 0.11, 0.35], [0.32, 0.59, 0, 0.04, 0.11, 0.35],
      ...[-0.29, 0.29].flatMap(x => [-0.29, 0.29].map(z => [x, 0.19, z, 0.027, 0.19, 0.027] as Box))] },
  { id: "dining_chair", asset: "models/dining_chair_02/model.gltf", position: [-0.84, 0, -1.12], rotation: Math.PI, massKg: 6,
    boxes: diningChair.map(shape => shape.bounds as Box) },
  { id: "tv_cabinet", asset: "models/modern_wooden_cabinet/model.gltf", position: [2.96, 0, 0.03], rotation: -Math.PI / 2, size: [1.60, 0.60, 0.50],
    boxes: [[0, 0.34, 0, 0.8, 0.26, 0.25], ...[-0.71, 0.71].flatMap(x => [-0.19, 0.19].map(z => [x, 0.04, z, 0.03, 0.04, 0.03] as Box))] },
  { id: "tv", asset: "generated/tv.gltf", position: [2.96, 0, 0.03], rotation: -Math.PI / 2, origin: "native",
    boxes: [[0, 1.12, 0, 0.635, 0.38, 0.0325], [0, 0.69, 0, 0.035, 0.07, 0.03], [0, 0.635, 0, 0.2, 0.015, 0.12]] },
  { id: "tv_plant", asset: "models/potted_plant_04/model.gltf", position: [2.81, 0.60, -0.65], size: tvPlant.size as Vec3,
    massKg: 1, friction: 0.55, boxes: tvPlant.shapes.map(shape => shape.bounds as Box) },
];

export const MOVABLE_FURNITURE = FURNITURE.filter(f => f.massKg !== undefined);
export const ENVIRONMENT_MESHES = [
  ...diningChair.map(shape => `<mesh name="dining_chair_${shape.name}" vertex="${shape.vertices.join(" ")}"/>`),
  ...tvPlant.shapes.map(shape => `<mesh name="tv_plant_${shape.name}" vertex="${shape.vertices.join(" ")}"/>`),
].join("");

const volume = (b: Box) => 8 * b[3] * b[4] * b[5];
const box = (name: string, values: Box, attributes = "") => `<geom name="${name}" type="box" pos="${values.slice(0, 3).join(" ")}" size="${values.slice(3).join(" ")}" contype="2" conaffinity="3" ${attributes}/>`;
export const ENVIRONMENT_BODIES = [
  ...ROOM_WALLS.map(wall => box(wall.id, [...wall.position, wall.width / 2, 1.35, 0.025],
    `quat="${Math.cos(wall.rotation / 2)} 0 ${Math.sin(wall.rotation / 2)} 0"`)),
  box("window", [-0.9, 1.65, -3.08, 0.96, 0.635, 0.04]),
  ...FURNITURE.map(f => {
    // Boxes describe Map bounds; these contact hulls come from the visible surfaces.
    const meshes = f.id === "dining_chair" ? diningChair : f.id === "tv_plant" ? tvPlant.shapes : undefined;
    const volumes = f.boxes.map((b, i) => meshes ? meshes[i].volume : volume(b));
    const totalVolume = volumes.reduce((sum, v) => sum + v, 0);
    const geoms = f.boxes.map((b, i) => {
      const attributes = f.massKg === undefined ? "" : `mass="${f.massKg * volumes[i] / totalVolume}" priority="1" friction="${f.friction ?? 0.35} 0.01 0.001"`;
      return meshes ? `<geom name="${f.id}_${i}" type="mesh" mesh="${f.id}_${meshes[i].name}" contype="2" conaffinity="3" ${attributes}/>` : box(`${f.id}_${i}`, b, attributes);
    }).join("");
    const door = f.id !== "fridge" ? "" : `<body name="fridge_door" pos="${FRIDGE_DOOR.hinge.join(" ")}">
      <joint name="fridge_hinge" type="hinge" axis="0 1 0" range="0 ${FRIDGE_DOOR.maxAngle}" damping="1" frictionloss="0.25" armature="0.01"/>
      <inertial pos="-0.41 0 0.022" mass="${FRIDGE_DOOR.massKg}" diaginertia="0.640 0.338 0.975"/>
      ${box("fridge_door_panel", [-0.41, 0, 0.022, 0.41, 0.559, 0.0225])}
      ${box("fridge_door_gasket", [-0.41, 0, 0, 0.41, 0.565, 0.009])}
      <geom name="fridge_handle" type="capsule" fromto="-0.7 -0.13 0.089 -0.7 0.1 0.089" size="0.012" contype="2" conaffinity="3" priority="1" friction="1.4 0.02 0.001"/>
      ${[-0.13, 0.1].map(y => `<geom type="capsule" fromto="-0.7 ${y} 0.059 -0.7 ${y} 0.089" size="0.012" contype="2" conaffinity="3"/>`).join("")}
      <site name="fridge_handle" pos="${FRIDGE_DOOR.handle.join(" ")}"/>
    </body>`;
    const kitchen = f.id !== "kitchen" ? "" : `<body name="kitchen_door" pos="${KITCHEN_DOOR.hinge.join(" ")}">
      <joint name="kitchen_hinge" type="hinge" axis="0 -1 0" range="0 ${KITCHEN_DOOR.maxAngle}" damping="0.7" frictionloss="0.15"/>
      <inertial pos="0.24 0 -0.02" mass="${KITCHEN_DOOR.massKg}" diaginertia="0.096 0.039 0.135"/>
      ${box("kitchen_door_panel", [0.24, 0, -0.02, 0.24, 0.38, 0.0125])}
      <geom name="kitchen_handle" type="capsule" fromto="0.13 0.33 0.024 0.35 0.33 0.024" size="0.008" contype="2" conaffinity="3" priority="1" friction="1.4 0.02 0.001"/>
      ${[0.13, 0.35].map(x => `<geom type="capsule" fromto="${x} 0.33 0 ${x} 0.33 0.024" size="0.008" contype="2" conaffinity="3"/>`).join("")}
      <site name="kitchen_handle" pos="${KITCHEN_DOOR.handle.join(" ")}"/>
    </body><body name="oven_door" pos="${OVEN_DOOR.hinge.join(" ")}">
      <joint name="oven_hinge" type="hinge" axis="1 0 0" range="0 ${OVEN_DOOR.maxAngle}" damping="1.5" frictionloss="0.35" stiffness="4.5" springref="0"/>
      <inertial pos="0 0.275 0.015" mass="${OVEN_DOOR.massKg}" diaginertia="0.071 0.075 0.145"/>
      ${box("oven_door_panel", [0, 0.2725, 0.015, 0.30, 0.2725, 0.015])}
      ${box("oven_window", [0, 0.295, 0.020, 0.235, 0.18, 0.0095])}
      <geom name="oven_handle" type="capsule" fromto="-0.22 0.48 0.059 0.22 0.48 0.059" size="0.011" contype="2" conaffinity="3" priority="1" friction="1.4 0.02 0.001"/>
      ${[-0.22, 0.22].map(x => `<geom type="capsule" fromto="${x} 0.48 0.023 ${x} 0.48 0.059" size="0.011" contype="2" conaffinity="3"/>`).join("")}
      <site name="oven_handle" pos="${OVEN_DOOR.handle.join(" ")}"/>
    </body>`;
    return `<body name="room_${f.id}" pos="${f.position.join(" ")}" quat="${Math.cos((f.rotation ?? 0) / 2)} 0 ${Math.sin((f.rotation ?? 0) / 2)} 0">${f.massKg === undefined ? "" : "<freejoint/>"}${geoms}${door}${kitchen}</body>`;
  }),
];
