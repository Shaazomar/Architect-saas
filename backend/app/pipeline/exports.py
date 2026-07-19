"""Stage 12 — Export AI.

Real exporters only:
- house.gltf   embedded-buffer glTF converted from the GLB byte-for-byte
               (preserves the injected lights and cameras)
- house.ifc    IFC4 via ifcopenshell: project/site/building/storey spatial
               tree, every mesh as the matching IFC class (walls, slabs,
               coverings, members, furnishing), rooms as IfcSpace volumes
- house.usdz   USD via usd-core: one UsdGeomMesh per element with
               UsdPreviewSurface materials, packaged with UsdUtils
- FBX          proprietary SDK format — declared pending, never faked

Meshes are vertex-welded before IFC/USD export (optimization); Draco /
meshopt GLB compression is declared pending.

Axis notes: the GLB is glTF Y-up; USD keeps Y-up; IFC is Z-up so vertices
map (x, y, z) -> (x, -z, y).
"""
from __future__ import annotations

import base64
import io
import json
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import trimesh

_IFC_CLASS = (
    ("wall", "IfcWall"),
    ("roof_parapet", "IfcWall"),
    ("roof", "IfcSlab"),
    ("slab", "IfcSlab"),
    ("skirting", "IfcCovering"),
    ("lintel", "IfcMember"),
    ("door_frame", "IfcMember"),
    ("room", "IfcSpace"),
    ("furniture", "IfcFurnishingElement"),
)


def glb_to_gltf_embedded(glb: bytes) -> bytes:
    """Convert a GLB to a single-file .gltf with a base64 data-URI buffer."""
    assert glb[:4] == b"glTF"
    json_len = struct.unpack("<I", glb[12:16])[0]
    doc = json.loads(glb[20 : 20 + json_len])
    offset = 20 + json_len
    if offset < len(glb):
        bin_len = struct.unpack("<I", glb[offset : offset + 4])[0]
        bin_data = glb[offset + 8 : offset + 8 + bin_len]
        if doc.get("buffers"):
            doc["buffers"][0]["uri"] = (
                "data:application/octet-stream;base64," + base64.b64encode(bin_data).decode()
            )
            doc["buffers"][0]["byteLength"] = len(bin_data)
    return json.dumps(doc).encode()


def _welded_meshes(glb: bytes) -> list[tuple[str, trimesh.Trimesh]]:
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    out = []
    for name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh = mesh.copy()
        mesh.merge_vertices()
        out.append((name, mesh))
    return out


def _ifc_class(name: str) -> str:
    for prefix, cls in _IFC_CLASS:
        if name.startswith(prefix):
            return cls
    return "IfcBuildingElementProxy"


def export_ifc(glb: bytes, project_name: str = "Architect SaaS Reconstruction") -> bytes:
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.geometry
    import ifcopenshell.api.project
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit

    f = ifcopenshell.api.project.create_file(version="IFC4")
    project = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name=project_name)
    ifcopenshell.api.unit.assign_unit(f)
    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        f, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=model,
    )

    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.root.create_entity(
        f, ifc_class="IfcBuildingStorey", name="Ground Floor"
    )
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(f, products=[building], relating_object=site)
    ifcopenshell.api.aggregate.assign_object(f, products=[storey], relating_object=building)

    for name, mesh in _welded_meshes(glb):
        # glTF Y-up -> IFC Z-up.
        verts = mesh.vertices[:, [0, 2, 1]].copy()
        verts[:, 1] *= -1.0
        element = ifcopenshell.api.root.create_entity(f, ifc_class=_ifc_class(name), name=name)
        rep = ifcopenshell.api.geometry.add_mesh_representation(
            f, context=body, vertices=[verts.tolist()], faces=[mesh.faces.tolist()]
        )
        ifcopenshell.api.geometry.assign_representation(f, product=element, representation=rep)
        if element.is_a("IfcSpace"):
            ifcopenshell.api.aggregate.assign_object(f, products=[element], relating_object=storey)
        else:
            ifcopenshell.api.spatial.assign_container(
                f, products=[element], relating_structure=storey
            )
    return f.to_string().encode()


def _base_color(mesh) -> tuple[np.ndarray, object] | None:
    mat = getattr(mesh.visual, "material", None)
    if mat is None or getattr(mat, "baseColorFactor", None) is None:
        return None
    rgba = np.asarray(mat.baseColorFactor, dtype=float)
    if rgba.max() > 1.0:
        rgba = rgba / 255.0
    return rgba, mat


def export_usdz(glb: bytes) -> bytes:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

    with tempfile.TemporaryDirectory() as tmp:
        usdc = str(Path(tmp) / "house.usdc")
        stage = Usd.Stage.CreateNew(usdc)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/House")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Scope.Define(stage, "/House/Materials")

        materials_cache: dict[str, UsdShade.Material] = {}

        def usd_material(rgba, mat) -> UsdShade.Material:
            key = re.sub(r"\W", "_", str(getattr(mat, "name", "mat")) or "mat")
            if key in materials_cache:
                return materials_cache[key]
            m = UsdShade.Material.Define(stage, f"/House/Materials/{key}")
            shader = UsdShade.Shader.Define(stage, f"/House/Materials/{key}/pbr")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                float(getattr(mat, "metallicFactor", 0.0) or 0.0)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                float(getattr(mat, "roughnessFactor", 0.8) or 0.8)
            )
            m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials_cache[key] = m
            return m

        for name, mesh in _welded_meshes(glb):
            safe = re.sub(r"\W", "_", name)
            prim = UsdGeom.Mesh.Define(stage, f"/House/{safe}")
            prim.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in mesh.vertices])
            prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            prim.CreateFaceVertexIndicesAttr([int(i) for i in mesh.faces.flatten()])
            colored = _base_color(mesh)
            if colored is not None:
                rgba, mat = colored
                UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(usd_material(rgba, mat))
                prim.CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])
        stage.Save()

        usdz_path = str(Path(tmp) / "house.usdz")
        if not UsdUtils.CreateNewUsdzPackage(usdc, usdz_path):
            raise RuntimeError("USDZ packaging failed")
        return Path(usdz_path).read_bytes()
