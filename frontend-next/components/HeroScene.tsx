"use client";
import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { inSphere } from "maath/random";

function useScrollN() {
  const ref = useRef(0);
  useEffect(() => {
    const on = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      ref.current = h > 0 ? window.scrollY / h : 0;
    };
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  return ref;
}

function Cloud({ count, radius, color, size }: { count: number; radius: number; color: string; size: number }) {
  const ref = useRef<THREE.Points>(null!);
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    inSphere(arr, { radius });
    return arr;
  }, [count, radius]);
  useFrame((_, dt) => {
    if (!ref.current) return;
    ref.current.rotation.y += dt * 0.03;
    ref.current.rotation.x += dt * 0.008;
  });
  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial transparent color={color} size={size} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} opacity={0.9} />
    </Points>
  );
}

function Core({ scrollN }: { scrollN: React.MutableRefObject<number> }) {
  const g = useRef<THREE.Group>(null!);
  const inner = useRef<THREE.Mesh>(null!);
  const ico = useMemo(() => new THREE.IcosahedronGeometry(1.7, 1), []);
  const wire = useMemo(() => new THREE.WireframeGeometry(ico), [ico]);
  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    if (g.current) {
      g.current.rotation.y += dt * 0.18;
      g.current.rotation.z = Math.sin(t * 0.2) * 0.15;
      const s = 1 + Math.sin(t * 1.1) * 0.03 + scrollN.current * 0.6;
      g.current.scale.setScalar(s);
    }
    if (inner.current) inner.current.rotation.x -= dt * 0.3;
  });
  return (
    <group ref={g}>
      <lineSegments geometry={wire}>
        <lineBasicMaterial color="#7c5cff" transparent opacity={0.55} blending={THREE.AdditiveBlending} />
      </lineSegments>
      <mesh ref={inner} geometry={ico}>
        <meshBasicMaterial color="#0a0f2a" transparent opacity={0.55} />
      </mesh>
      <points geometry={ico}>
        <pointsMaterial color="#28e0e0" size={0.05} sizeAttenuation transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
    </group>
  );
}

function Ring({ r, tilt, color, speed }: { r: number; tilt: number; color: string; speed: number }) {
  const ref = useRef<THREE.Mesh>(null!);
  const geo = useMemo(() => new THREE.TorusGeometry(r, 0.006, 8, 160), [r]);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.z += dt * speed;
  });
  return (
    <mesh ref={ref} geometry={geo} rotation={[tilt, tilt * 0.6, 0]}>
      <meshBasicMaterial color={color} transparent opacity={0.5} blending={THREE.AdditiveBlending} />
    </mesh>
  );
}

function Rig({ scrollN }: { scrollN: React.MutableRefObject<number> }) {
  const { camera } = useThree();
  const mouse = useRef({ x: 0, y: 0 });
  useEffect(() => {
    const on = (e: MouseEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.current.y = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener("mousemove", on);
    return () => window.removeEventListener("mousemove", on);
  }, []);
  useFrame(() => {
    const tx = mouse.current.x * 1.2;
    const ty = -mouse.current.y * 1.0 + scrollN.current * 2.5;
    camera.position.x += (tx - camera.position.x) * 0.045;
    camera.position.y += (ty - camera.position.y) * 0.045;
    camera.position.z = 9 - scrollN.current * 2.2;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

export default function HeroScene() {
  const scrollN = useScrollN();
  const [dpr, setDpr] = useState(1.2);
  useEffect(() => setDpr(Math.min(window.devicePixelRatio || 1, 2)), []);
  return (
    <Canvas
      dpr={dpr}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ position: [0, 0, 9], fov: 46 }}
      style={{ position: "absolute", inset: 0 }}
    >
      <Rig scrollN={scrollN} />
      <Core scrollN={scrollN} />
      <Ring r={2.6} tilt={1.2} color="#28e0e0" speed={0.25} />
      <Ring r={3.3} tilt={-0.7} color="#ff4d8d" speed={-0.18} />
      <Ring r={4.1} tilt={0.5} color="#7c5cff" speed={0.12} />
      <Cloud count={2600} radius={9} color="#2aa8ff" size={0.02} />
      <Cloud count={1500} radius={6.5} color="#7c5cff" size={0.03} />
      <Cloud count={700} radius={11} color="#ff4d8d" size={0.03} />
      <EffectComposer>
        <Bloom intensity={1.15} luminanceThreshold={0.08} luminanceSmoothing={0.9} mipmapBlur radius={0.75} />
      </EffectComposer>
    </Canvas>
  );
}
