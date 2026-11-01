/* eslint-disable react/immutability -- The renderer follows the measured MuJoCo joint. */
import { Suspense, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { Mesh } from "three";
import type { Simulation } from "../simulation/mujoco.ts";

export function Fridge({ sim }: { sim: Simulation }) {
  return <Suspense fallback={null}><LoadedFridge sim={sim}/></Suspense>;
}

function LoadedFridge({ sim }: { sim: Simulation }) {
  const { scene } = useGLTF("/assets/realistic-home-v1/generated/fridge.gltf");
  const model = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse(o => { if (o instanceof Mesh) { o.castShadow = true; o.receiveShadow = true; } });
    return clone;
  }, [scene]);
  const door = useMemo(() => model.getObjectByName("fridge_door")!, [model]);
  useFrame(() => { door.rotation.y = sim.q("fridge_hinge"); }, -0.5);
  return <primitive object={model}/>;
}
