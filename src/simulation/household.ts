import { TABLE_HEIGHT } from "./table.ts";

// Each object is one free body made from several primitive collision shapes.
// Segmented walls keep the mug and plate interiors open, including the handle.
function ring(radius: number, halfThickness: number, halfHeight: number, y: number, mass: number) {
  const count = 24, tangentHalf = radius * Math.tan(Math.PI / count) * 1.02;
  return Array.from({ length: count }, (_, i) => {
    const angle = i * Math.PI * 2 / count;
    return `<geom type="box" pos="${radius * Math.cos(angle)} ${y} ${radius * Math.sin(angle)}"
      size="${halfThickness} ${halfHeight} ${tangentHalf}" euler="0 ${-angle} 0" mass="${mass / count}"/>`;
  });
}

const mugGeoms = [
  '<geom type="cylinder" pos="0 -0.056 0" size="0.05 0.007" quat="0.707107 0.707107 0 0" mass="0.07" rgba="0.74 0.42 0.28 1" contype="2" conaffinity="3"/>',
  ...ring(0.05, 0.005, 0.06, 0, 0.19).map(g => g.replace('/>', 'rgba="0.82 0.55 0.39 1" contype="2" conaffinity="3"/>')),
  ...[-0.038, 0.038].map(y => `<geom type="box" pos="0.078 ${y} 0" size="0.024 0.006 0.009" mass="0.01" rgba="0.82 0.55 0.39 1" contype="2" conaffinity="3"/>`),
  '<geom type="box" pos="0.1 0 0" size="0.006 0.038 0.009" mass="0.02" rgba="0.82 0.55 0.39 1" contype="2" conaffinity="3"/>',
].map((geom, i) => {
  const name = i === 0 ? "mug_bottom" : `mug_${i}`;
  return { name, xml: geom.replace("<geom", `<geom name="${name}"`) };
});

// Cup/floor compliance preserves the basket's table support and robot contact response.
export const MUG_FLOOR_CONTACTS = mugGeoms.map(({ name }) =>
  `<pair geom1="${name}" geom2="basket_0" condim="4" friction="0.65 0.65 0.01 0.001 0.001" solref="0.012 1" solimp="0.99 0.999 0.001 0.5 2"/>`).join("");

export const HOUSEHOLD_BODIES = [
  `<body name="mug" pos="0 ${TABLE_HEIGHT + 0.31} 0.44"><freejoint/>`,
  ...mugGeoms.map(({ xml }) => xml),
  '</body>',
  `<body name="bottle" pos="-0.22 ${TABLE_HEIGHT + 0.38} -0.25"><freejoint/>`,
  '<geom type="cylinder" pos="0 -0.015 0" size="0.045 0.095" quat="0.707107 0.707107 0 0" mass="0.35" rgba="0.24 0.47 0.39 1" contype="2" conaffinity="3"/>',
  '<geom type="capsule" pos="0 0.073 0" size="0.035 0.013" quat="0.707107 0.707107 0 0" mass="0.05" rgba="0.24 0.47 0.39 1" contype="2" conaffinity="3"/>',
  '<geom type="cylinder" pos="0 0.119 0" size="0.019 0.03" quat="0.707107 0.707107 0 0" mass="0.03" rgba="0.24 0.47 0.39 1" contype="2" conaffinity="3"/>',
  '<geom type="cylinder" pos="0 0.149 0" size="0.022 0.009" quat="0.707107 0.707107 0 0" mass="0.02" rgba="0.19 0.25 0.22 1" contype="2" conaffinity="3"/>',
  '<geom type="cylinder" pos="0 -0.012 0" size="0.0452 0.041" quat="0.707107 0.707107 0 0" mass="0.005" rgba="0.88 0.85 0.69 1" contype="2" conaffinity="3"/>',
  '</body>',
  `<body name="plate" pos="0.09 ${TABLE_HEIGHT + 0.30} -0.24"><freejoint/>`,
  '<geom type="cylinder" pos="0 -0.008 0" size="0.105 0.006" quat="0.707107 0.707107 0 0" mass="0.18" rgba="0.91 0.92 0.86 1" contype="2" conaffinity="3"/>',
  ...ring(0.1, 0.007, 0.008, 0.005, 0.07).map(g => g.replace('/>', 'rgba="0.66 0.75 0.69 1" contype="2" conaffinity="3"/>')),
  '</body>',
];

// Three distinct sealed drinks, with body / shoulder / neck / cap contact shapes.
export const DRINK_BODIES = [
  { id: "tea", position: [2.34, 0.52, -2.39], radius: 0.032, bottom: -0.11, shoulder: 0.055, neck: 0.092, cap: 0.114, mass: 0.55 },
  { id: "juice", position: [2.55, 0.88, -2.39], radius: 0.03, bottom: -0.095, shoulder: 0.045, neck: 0.084, cap: 0.104, mass: 0.42 },
  { id: "milk", position: [2.78, 0.89, -2.39], radius: 0.0335, bottom: -0.095, shoulder: 0.06, neck: 0.09, cap: 0.108, mass: 0.36 },
].map(d => `<body name="${d.id}" pos="${d.position.join(" ")}"><freejoint/>
  <geom type="cylinder" pos="0 ${(d.bottom + d.shoulder) / 2} 0" size="${d.radius} ${(d.shoulder - d.bottom) / 2}" quat="0.707107 0.707107 0 0" mass="${d.mass * 0.85}" contype="2" conaffinity="3"/>
  <geom type="capsule" pos="0 ${d.shoulder + 0.006} 0" size="${d.radius * 0.72} 0.005" quat="0.707107 0.707107 0 0" mass="${d.mass * 0.08}" contype="2" conaffinity="3"/>
  <geom type="cylinder" pos="0 ${d.neck} 0" size="0.015 0.018" quat="0.707107 0.707107 0 0" mass="${d.mass * 0.04}" contype="2" conaffinity="3"/>
  <geom type="cylinder" pos="0 ${d.cap} 0" size="0.018 0.008" quat="0.707107 0.707107 0 0" mass="${d.mass * 0.03}" contype="2" conaffinity="3"/>
</body>`);
