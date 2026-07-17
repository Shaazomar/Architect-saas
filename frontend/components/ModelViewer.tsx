"use client";

import { Suspense, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, useGLTF } from "@react-three/drei";

function Model({ url, showRoof }: { url: string; showRoof: boolean }) {
  const { scene } = useGLTF(url);
  useEffect(() => {
    scene.traverse((obj) => {
      if (obj.name.startsWith("roof_")) obj.visible = showRoof;
    });
  }, [scene, showRoof]);
  return <primitive object={scene} />;
}

export default function ModelViewer({ url, showRoof = false }: { url: string; showRoof?: boolean }) {
  return (
    <Canvas camera={{ position: [18, 20, 18], fov: 45 }} shadows>
      <color attach="background" args={["#0a0a0a"]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[15, 25, 10]} intensity={1.4} castShadow />
      <hemisphereLight args={["#cbd5e1", "#1c1917", 0.5]} />
      <Suspense fallback={null}>
        <Model url={url} showRoof={showRoof} />
      </Suspense>
      <Grid
        args={[100, 100]}
        position={[0, -0.11, 0]}
        cellColor="#1f2937"
        sectionColor="#374151"
        infiniteGrid
        fadeDistance={80}
      />
      <OrbitControls makeDefault target={[10, 0, 8]} />
    </Canvas>
  );
}
