"""A GLB made small for the web: decimated, its textures scaled down, Draco and JPEG.
blender -b -P glb_small.py -- SRC DST [TRIANGLES] [TEXTURE_PX]"""
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
triangles = int(argv[2]) if len(argv) > 2 else 40000
texture = int(argv[3]) if len(argv) > 3 else 1024

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
total = sum(len(o.data.polygons) for o in meshes) or 1
ratio = min(1.0, triangles / total)
for o in meshes:
    if ratio < 1.0:
        m = o.modifiers.new("decimate", "DECIMATE")
        m.ratio = ratio
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=m.name)
for img in bpy.data.images:
    w, h = img.size
    if max(w, h) > texture:
        s = texture / max(w, h)
        img.scale(max(1, int(w * s)), max(1, int(h * s)))
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_draco_mesh_compression_enable=True,
                          export_image_format="JPEG", export_jpeg_quality=82)
print(f"glb_small: {total} -> {sum(len(o.data.polygons) for o in meshes)} faces")
