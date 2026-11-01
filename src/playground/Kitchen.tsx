/* eslint-disable react/immutability -- The renderer follows the measured MuJoCo joints. */
import { Suspense, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { Mesh } from "three";
import type { Simulation } from "../simulation/mujoco.ts";

export function Kitchen({ sim }: { sim: Simulation }) {
  return <Suspense fallback={null}><LoadedKitchen sim={sim}/></Suspense>;
}

function LoadedKitchen({ sim }: { sim: Simulation }) {
  const { scene } = useGLTF("/assets/realistic-home-v1/generated/kitchen.gltf");
  const model = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse(o => { if (o instanceof Mesh) { o.castShadow = true; o.receiveShadow = true; } });
    return clone;
  }, [scene]);
  const cabinet = useMemo(() => model.getObjectByName("kitchen_door")!, [model]);
  const oven = useMemo(() => model.getObjectByName("oven_door")!, [model]);
  useFrame(() => { cabinet.rotation.y = -sim.q("kitchen_hinge"); oven.rotation.x = sim.q("oven_hinge"); }, -0.5);
  return <primitive object={model}/>;
}
