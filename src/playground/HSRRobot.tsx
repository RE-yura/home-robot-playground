/* eslint-disable react/immutability -- Robot mesh transforms follow MuJoCo's measured body poses. */
import { Suspense, useMemo, useRef } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { Group, Loader, Matrix4, Mesh } from "three";
import { HSROBJLoader } from "../robot/hsr-mesh.ts";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import type { Simulation } from "../simulation/mujoco.ts";
import { HSR_LINKS, rpyQuaternion } from "../robot/hsr-model.ts";

// Cache by URL through a stable loader class; memoized loader instances are lost on initial Suspense retries.
class HSRMeshLoader extends Loader<Group> {
  override load(url: string, onLoad: (group: Group) => void, onProgress?: (event: ProgressEvent) => void, onError?: (error: unknown) => void) {
    new MTLLoader(this.manager).load(url.replace(/\.obj$/, ".mtl"), materials => {
      new HSROBJLoader(this.manager).setMaterials(materials).load(url, onLoad, onProgress, onError);
    }, onProgress, onError);
  }
}

// Original asset files are unchanged; MTL colors and textures use the room's PBR lighting.
// ROS Z-up / +X forward → Y-up / -Z forward.
function OriginalMesh({ path }: { path: string }) {
  return <Suspense fallback={null}><LoadedOriginalMesh path={path}/></Suspense>;
}

function LoadedOriginalMesh({ path }: { path: string }) {
  const object = useLoader(HSRMeshLoader, `/assets/hsr-original-v1/meshes/${path}.obj`);
  const clone = useMemo(() => {
    const c = object.clone(true);
    c.traverse(o => { if (o instanceof Mesh) { o.castShadow = true; o.receiveShadow = true; } });
    return c;
  }, [object]);
  return <primitive object={clone}/>;
}

export function HSRRobot({ sim }: { sim: Simulation }) {
  const groups = useRef<(Group | null)[]>([]), matrix = useMemo(() => new Matrix4(), []);
  const links = useMemo(() => [{ name: "hsr_frame", visuals: ["base_v2/base_light", "base_v2/body_light", "base_v2/torso_base"].map(path =>
    ({ type: "mesh", path, xyz: [0, 0, 0], rpy: [0, 0, 0] })) }, ...HSR_LINKS], []);
  const bodies = useMemo(() => links.map(link => sim.id("BODY", link.name)), [sim, links]);
  useFrame(() => {
    bodies.forEach((body, i) => {
      const group = groups.current[i]; if (!group) return;
      const p = sim.data.xpos, r = sim.data.xmat, k = body * 9;
      group.position.set(p[body * 3], p[body * 3 + 1], p[body * 3 + 2]);
      matrix.set(r[k], r[k + 1], r[k + 2], 0, r[k + 3], r[k + 4], r[k + 5], 0,
        r[k + 6], r[k + 7], r[k + 8], 0, 0, 0, 0, 1);
      group.quaternion.setFromRotationMatrix(matrix);
    });
  }, -0.5);
  return <>{links.map((link, i) => <group key={link.name} name={`robot:${link.name}`} ref={g => { groups.current[i] = g; }}>
    {link.visuals.map((shape, j) => <group key={j} position={shape.xyz as [number, number, number]} quaternion={rpyQuaternion(shape.rpy)}>
      {shape.type === "mesh" ? <OriginalMesh path={shape.path!}/> : <mesh rotation={[Math.PI / 2, 0, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.017, 0.017, 0.02, 24]}/><meshStandardMaterial color="#c4c4c4" roughness={0.4}/>
      </mesh>}
    </group>)}
    {link.name === "head_tilt_link" && <>
      {[0.07, -0.07].map(y => <group key={y} position={[0.02, y, 0.061]} quaternion={[-0.5, 0.5, -0.5, 0.5]}><OriginalMesh path="sensors/head_camera"/></group>)}
      <group position={[-0.079796, 0.022, 0.215172]} quaternion={[-0.5, 0.5, -0.5, 0.5]}><OriginalMesh path="sensors/rgbd_light"/></group>
    </>}
  </group>)}</>;
}
