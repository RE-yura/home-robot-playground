import { FURNITURE, ROOM_WALLS } from "../simulation/environment.ts";
import { Suspense, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Environment } from "@react-three/drei";
import type { Group } from "three";
import type { Simulation } from "../simulation/mujoco.ts";
import { AssetModel, PbrMaterial } from "./VisualAssets.tsx";
import { Fridge } from "./Fridge.tsx";
import { Kitchen } from "./Kitchen.tsx";

// Detailed rendering is independent of the small, stable MuJoCo collision model.
export function RoomEnvironment({ sim }: { sim: Simulation }) {
  const groups = useRef<(Group | null)[]>([]);
  const bodies = useMemo(() => FURNITURE.map(f => f.massKg === undefined ? -1 : sim.id("BODY", `room_${f.id}`)), [sim]);
  useFrame(() => {
    const p = sim.data.xpos, r = sim.data.xquat;
    groups.current.forEach((group, i) => {
      const body = bodies[i];
      if (!group || body < 0) return;
      group.position.set(p[body * 3], p[body * 3 + 1], p[body * 3 + 2]);
      group.quaternion.set(r[body * 4 + 1], r[body * 4 + 2], r[body * 4 + 3], r[body * 4]);
    });
  }, -0.5);
  return <group>
    <Suspense fallback={null}><Environment files="/assets/realistic-home-v1/lighting/studio.hdr" environmentIntensity={0.65}/></Suspense>
    <mesh position={[0, 0.0005, -0.2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[6.6, 5.9]}/><PbrMaterial asset="laminate_floor_02" repeat={[3.88, 3.47]}/>
    </mesh>
    {ROOM_WALLS.map(wall => <group key={wall.id} position={wall.position} rotation={[0, wall.rotation, 0]}>
      <mesh castShadow receiveShadow>
        <boxGeometry args={[wall.width, 2.7, 0.05]}/><PbrMaterial asset="terlenka" repeat={[wall.width / 0.3, 9]}/>
      </mesh>
    </group>)}
    <group position={[-0.9, 1.65, -3.11]}>
      <mesh><boxGeometry args={[1.92, 1.27, 0.06]}/><meshStandardMaterial color="#e9e8df" roughness={0.28}/></mesh>
      <mesh position={[0, 0, 0.034]}><planeGeometry args={[1.78, 1.12]}/><meshStandardMaterial color="#c4dce3" emissive="#dbeef1" emissiveIntensity={0.5} roughness={0.08} metalness={0.25}/></mesh>
      <mesh position={[0, 0, 0.053]}><boxGeometry args={[0.032, 1.13, 0.022]}/><meshStandardMaterial color="#f4f1e8" roughness={0.3}/></mesh>
      <mesh position={[0, 0, 0.053]}><boxGeometry args={[1.8, 0.032, 0.022]}/><meshStandardMaterial color="#f4f1e8" roughness={0.3}/></mesh>
      <mesh position={[0, -0.64, 0.04]} castShadow receiveShadow><boxGeometry args={[2.03, 0.045, 0.16]}/><PbrMaterial asset="wood_table_001" repeat={[2, 0.4]}/></mesh>
    </group>
    {FURNITURE.map((f, i) => <group key={f.id} name={`furniture:${f.id}`} ref={g => { groups.current[i] = g; }} position={f.position} rotation={[0, f.rotation ?? 0, 0]}>
      {f.id === "fridge" ? <Fridge sim={sim}/> : f.id === "kitchen" ? <Kitchen sim={sim}/> : <AssetModel url={`/assets/realistic-home-v1/${f.asset}`} size={f.size} origin={f.origin}/>}
    </group>)}
    <mesh position={[-2.14, 0.002, -0.65]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[1.95, 2.32]}/><PbrMaterial asset="fabric_pattern_05" repeat={[5, 5]} color="#c3c0ab"/>
    </mesh>
    <AssetModel url="/assets/realistic-home-v1/models/potted_plant_04/model.gltf" position={[-1.57, 0.77, -2.68]} size={[0.28, 0.44, 0.3]}/>
    <AssetModel url="/assets/realistic-home-v1/generated/table.gltf" origin="native"/>
  </group>;
}
