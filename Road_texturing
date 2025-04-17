bl_info = {
    "name": "Road UV Mapping (Manual + Auto Ratio) with Separate Materials, Default Angle=90",
    "author": "OpenAI / ChatGPT Example",
    "version": (1, 0),
    "blender": (3, 5, 0),
    "category": "Object",
    "description": "Apply road/intersection textures with shape-based or manual tagging, ensuring separate materials for roads vs. intersections. Default angle=90° on gather.",
}

import bpy
import math
from mathutils import Matrix, Vector

# ------------------------------------------------------------
# 1) PCA + OBB HELPER FUNCTIONS
# ------------------------------------------------------------
def jacobi_eigen_decomposition(A, max_iterations=50):
    a = Matrix(A)
    v = Matrix.Identity(3)

    for _ in range(max_iterations):
        a01 = abs(a[0][1])
        a02 = abs(a[0][2])
        a12 = abs(a[1][2])

        p, q = 0, 1
        max_val = a01
        if a02 > max_val:
            p, q = 0, 2
            max_val = a02
        if a12 > max_val:
            p, q = 1, 2
            max_val = a12

        if max_val < 1e-12:
            break

        if abs(a[p][p] - a[q][q]) < 1e-12:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan2(2*a[p][q], a[q][q] - a[p][p])

        c = math.cos(phi)
        s = math.sin(phi)

        J = Matrix.Identity(3)
        J[p][p] = c
        J[q][q] = c
        J[p][q] = s
        J[q][p] = -s

        a = J.transposed() @ a @ J
        v = v @ J

    eigenvalues = (a[0][0], a[1][1], a[2][2])
    eigenvectors = [v.col[i] for i in range(3)]
    return eigenvalues, eigenvectors


def compute_pca_obb(points):
    """ Returns (center, rotation, extents) from PCA OBB. """
    if not points:
        return (Vector((0,0,0)), Matrix.Identity(3), Vector((0,0,0)))

    center = Vector((0,0,0))
    for pt in points:
        center += pt
    center /= len(points)

    # Covariance
    xx = xy = xz = yy = yz = zz = 0.0
    n = float(len(points))
    for pt in points:
        r = pt - center
        xx += r.x * r.x
        xy += r.x * r.y
        xz += r.x * r.z
        yy += r.y * r.y
        yz += r.y * r.z
        zz += r.z * r.z

    xx /= n
    xy /= n
    xz /= n
    yy /= n
    yz /= n
    zz /= n

    cov = Matrix(((xx, xy, xz),
                  (xy, yy, yz),
                  (xz, yz, zz)))

    eigenvals, eigenvecs = jacobi_eigen_decomposition(cov)

    # Sort by descending
    sorted_indices = sorted(range(3), key=lambda i: eigenvals[i], reverse=True)
    evecs_sorted = [eigenvecs[i] for i in sorted_indices]

    rot = Matrix((evecs_sorted[0], evecs_sorted[1], evecs_sorted[2])).transposed()

    # Extents
    inv_rot = rot.inverted()
    local_pts = [inv_rot @ (p - center) for p in points]

    min_corner = local_pts[0].copy()
    max_corner = local_pts[0].copy()
    for lp in local_pts[1:]:
        min_corner.x = min(min_corner.x, lp.x)
        min_corner.y = min(min_corner.y, lp.y)
        min_corner.z = min(min_corner.z, lp.z)
        max_corner.x = max(max_corner.x, lp.x)
        max_corner.y = max(max_corner.y, lp.y)
        max_corner.z = max(max_corner.z, lp.z)

    local_center = 0.5*(min_corner + max_corner)
    extents = 0.5*(max_corner - min_corner)
    world_center = center + rot @ local_center

    return (world_center, rot, extents)


def apply_road_uvs_one_tile_across_width(obj, u_scale=10.0, texture_rotation_degrees=0.0):
    """
    Creates a "RoadUV" map where:
      U = dot along largest PCA axis / u_scale
      V = 0..1 across second largest axis
      Then rotate (u,v) in 2D by texture_rotation_degrees if needed.
    """
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="RoadUV")
    uv_layer = mesh.uv_layers.active

    # Gather points in world space
    wpoints = [obj.matrix_world @ v.co for v in mesh.vertices]
    if not wpoints:
        print(f"[apply_road_uvs] {obj.name} has no vertices.")
        return

    obb_center, obb_rot, _ = compute_pca_obb(wpoints)

    x_axis = obb_rot.col[0].normalized()
    y_axis = obb_rot.col[1].normalized()

    y_vals = [(p - obb_center).dot(y_axis) for p in wpoints]
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_rng = (y_max - y_min) if (y_max>y_min) else 1e-9

    angle = math.radians(texture_rotation_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def rotate_uv(u, v):
        u2 = cos_a*u - sin_a*v
        v2 = sin_a*u + cos_a*v
        return (u2, v2)

    # Build uv
    uv_cache = [None]*len(mesh.vertices)
    for i, vtx in enumerate(mesh.vertices):
        wp = wpoints[i]
        local = wp - obb_center
        u_val = local.dot(x_axis)/u_scale
        valY = local.dot(y_axis)
        v_val = (valY - y_min)/y_rng

        (u_rot, v_rot) = rotate_uv(u_val, v_val)
        uv_cache[i] = (u_rot, v_rot)

    # Assign to loops
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = uv_cache[vi]

    mesh.update()


def assign_bitmap_to_object(obj, image_path, material_name="RoadTextureMaterial"):
    """
    Creates or reuses a material with name 'material_name',
    loads image_path, and assigns it to 'obj'.
    """
    if material_name in bpy.data.materials:
        mat = bpy.data.materials[material_name]
    else:
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    # Clear nodes
    for node in nodes:
        nodes.remove(node)

    # Principled + Image
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.location = (-300, 0)
    try:
        img = bpy.data.images.load(image_path)
        tex_node.image = img
        print(f"[assign_bitmap_to_object] Loaded '{image_path}' into '{material_name}'")
    except:
        print(f"[assign_bitmap_to_object] FAILED to load '{image_path}'")

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (200, 0)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # Assign
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


# ------------------------------------------------------------
# 2) DATA STRUCTURE FOR MANUAL STRATEGY
# ------------------------------------------------------------
class RoadIntersectionItem(bpy.types.PropertyGroup):
    """
    Holds per-object intersection info for manual strategy.
    - is_intersection: if True, we'll assign intersection_blank.png
    - angle_degrees: rotation of UV
    - u_scale: how many units = 1 tile
    """
    obj_name:  bpy.props.StringProperty(name="Object Name", default="")
    is_intersection: bpy.props.BoolProperty(name="Is Intersection", default=False)
    # Default angle=90
    angle_degrees: bpy.props.FloatProperty(name="Angle", default=90.0)
    u_scale: bpy.props.FloatProperty(name="U-Scale", default=10.0)


def get_object_by_name(name):
    """Helper: returns the object with a given name or None."""
    return bpy.data.objects.get(name, None)

# ------------------------------------------------------------
# 3) OPERATORS FOR MANUAL + SHAPE MARK
# ------------------------------------------------------------

class ROAD_OT_GatherSelected(bpy.types.Operator):
    """
    Gather selected MESH objects into the list
    so we can decide if they're intersections or normal roads.
    Sets default angle_degrees=90 for each newly added item.
    """
    bl_idname = "road.gather_selected"
    bl_label = "Gather Selected"

    def execute(self, context):
        scene = context.scene
        scene.road_intersections.clear()

        for obj in context.selected_objects:
            if obj.type == 'MESH':
                item = scene.road_intersections.add()
                item.obj_name = obj.name
                item.is_intersection = False
                # item.angle_degrees = 90.0  # Not needed if default=90 is set in the PropertyGroup
                item.u_scale = 10.0
        return {'FINISHED'}


class ROAD_OT_MarkByShapeRatio(bpy.types.Operator):
    """
    For each item in the list, compute shape ratio and mark is_intersection if ratio < threshold.
    """
    bl_idname = "road.mark_by_shape_ratio"
    bl_label = "Mark by Shape Ratio"

    def execute(self, context):
        scene = context.scene
        ratio_threshold = scene.mark_ratio_threshold

        for item in scene.road_intersections:
            obj = get_object_by_name(item.obj_name)
            if not obj or obj.type != 'MESH':
                continue

            # Compute shape ratio
            wpoints = [obj.matrix_world @ v.co for v in obj.data.vertices]
            _, _, exts = compute_pca_obb(wpoints)
            arr = sorted([abs(exts.x), abs(exts.y), abs(exts.z)], reverse=True)
            largest = arr[0]
            second = arr[1] if len(arr)>1 else 1e-9
            ratio = largest/second if second!=0 else 999999

            # If ratio < threshold => more "square" => intersection
            if ratio < ratio_threshold:
                item.is_intersection = True
            else:
                item.is_intersection = False

        return {'FINISHED'}


class ROAD_OT_ApplyManualStrategy(bpy.types.Operator):
    """
    Apply Road or Intersection textures to each item in the list,
    but only if the item is still selected in the 3D view,
    using distinct material names to avoid overwriting.
    """
    bl_idname = "road.apply_manual_strategy"
    bl_label = "Apply UV/Textures (Manual Strategy)"

    def execute(self, context):
        scene = context.scene
        # Texture paths for roads & intersections
        road_tex = r"D:\OneDrive - GODREJ INDUSTRIES LIMITED\Desktop\GPL-projects\Contour\Doddaballapur\Visualisation\Textures\road.png"
        intersection_tex = r"D:\OneDrive - GODREJ INDUSTRIES LIMITED\Desktop\GPL-projects\Contour\Doddaballapur\Visualisation\Textures\intersection_blank.png"

        for item in scene.road_intersections:
            obj = get_object_by_name(item.obj_name)
            if not obj or obj.type != 'MESH':
                continue

            # Only process if the object is currently selected
            if not obj.select_get():
                continue

            # Distinct material names => no overwriting
            if item.is_intersection:
                tex_path = intersection_tex
                mat_name = "IntersectionMaterial"
            else:
                tex_path = road_tex
                mat_name = "RoadMaterial"

            # OBB-based UV
            apply_road_uvs_one_tile_across_width(
                obj,
                u_scale=item.u_scale,
                texture_rotation_degrees=item.angle_degrees
            )

            # Assign the material
            assign_bitmap_to_object(obj, tex_path, material_name=mat_name)

        return {'FINISHED'}

# ------------------------------------------------------------
# 4) AUTO RATIO STRATEGY: "DO IT ALL" BUTTON
# ------------------------------------------------------------

class ROAD_OT_ApplyAutoRatioStrategy(bpy.types.Operator):
    """Processes all SELECTED meshes in a single pass using a global ratio threshold, angle, and scale, with distinct materials."""
    bl_idname = "road.apply_auto_ratio"
    bl_label = "Apply UV/Textures (Auto Ratio Strategy)"

    def execute(self, context):
        scene = context.scene
        # Hardcoded or user-set texture paths for roads & intersections
        road_tex = r"D:\OneDrive - GODREJ INDUSTRIES LIMITED\Desktop\GPL-projects\Contour\Doddaballapur\Visualisation\Textures\road.png"
        intersection_tex = r"D:\OneDrive - GODREJ INDUSTRIES LIMITED\Desktop\GPL-projects\Contour\Doddaballapur\Visualisation\Textures\intersection_blank.png"

        ratio_threshold = scene.auto_ratio_threshold
        global_angle = scene.auto_global_angle
        global_scale = scene.auto_global_scale

        selected_meshes = [o for o in context.selected_objects if o.type=='MESH']
        for obj in selected_meshes:
            wpoints = [obj.matrix_world @ v.co for v in obj.data.vertices]
            _, _, exts = compute_pca_obb(wpoints)

            arr = sorted([abs(exts.x), abs(exts.y), abs(exts.z)], reverse=True)
            largest = arr[0]
            second = arr[1] if len(arr) > 1 else 1e-9
            shape_ratio = largest / second if second != 0 else 999999

            if shape_ratio > ratio_threshold:
                # Considered a "Road"
                mat_name = "RoadMaterial"
                tex_path = road_tex
                apply_road_uvs_one_tile_across_width(obj,
                    u_scale=global_scale,
                    texture_rotation_degrees=global_angle
                )
            else:
                # Considered an "Intersection"
                mat_name = "IntersectionMaterial"
                tex_path = intersection_tex
                apply_road_uvs_one_tile_across_width(obj,
                    u_scale=global_scale,
                    texture_rotation_degrees=0.0
                )

            assign_bitmap_to_object(obj, tex_path, material_name=mat_name)

        return {'FINISHED'}

# ------------------------------------------------------------
# 5) UI PANELS
# ------------------------------------------------------------

class ROAD_PT_IntersectionPanel(bpy.types.Panel):
    """UI panel for the manual tagging approach (plus auto shape marking)."""
    bl_label = "Road Intersections (Manual + Shape Mark)"
    bl_idname = "ROAD_PT_intersection_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Road Intersections"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.operator("road.gather_selected", text="Gather Selected")

        # "Mark by Shape Ratio" controls
        row2 = layout.row()
        row2.prop(scene, "mark_ratio_threshold", text="Threshold")
        row2.operator("road.mark_by_shape_ratio", text="Mark by Shape Ratio")

        # List items
        for i, item in enumerate(scene.road_intersections):
            box = layout.box()
            row0 = box.row()
            row0.prop(item, "obj_name", text="Object")
            row1 = box.row()
            row1.prop(item, "is_intersection", text="IsInt?")
            row1.prop(item, "angle_degrees", text="Angle")
            row1.prop(item, "u_scale", text="U Scale")

        layout.operator("road.apply_manual_strategy", text="Apply UV/Textures (Manual Strategy)")


class ROAD_PT_AutoRatioPanel(bpy.types.Panel):
    """Panel for the automatic shape ratio strategy."""
    bl_label = "Auto Ratio Strategy"
    bl_idname = "ROAD_PT_auto_ratio_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Road Intersections"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text="Automatic Intersection Detection by Ratio")

        row = layout.row()
        row.prop(scene, "auto_ratio_threshold", text="Ratio Threshold")

        row2 = layout.row()
        row2.prop(scene, "auto_global_scale", text="U Scale")
        row2.prop(scene, "auto_global_angle", text="Angle")

        layout.operator("road.apply_auto_ratio", text="Apply UV/Textures (Auto Ratio Strategy)")

# ------------------------------------------------------------
# 6) REGISTER
# ------------------------------------------------------------

classes = (
    RoadIntersectionItem,
    ROAD_OT_GatherSelected,
    ROAD_OT_MarkByShapeRatio,
    ROAD_OT_ApplyManualStrategy,
    ROAD_OT_ApplyAutoRatioStrategy,
    ROAD_PT_IntersectionPanel,
    ROAD_PT_AutoRatioPanel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.road_intersections = bpy.props.CollectionProperty(type=RoadIntersectionItem)

    # For manual shape marking
    bpy.types.Scene.mark_ratio_threshold = bpy.props.FloatProperty(
        name="Ratio Threshold", default=3.0, min=0.01,
        description="If largest/second < threshold => intersection"
    )

    # For auto ratio approach
    bpy.types.Scene.auto_ratio_threshold = bpy.props.FloatProperty(
        name="auto_ratio_threshold", default=3.0, min=1.0,
        description="Used by the auto ratio strategy (If shape_ratio > this => road, else intersection)"
    )
    bpy.types.Scene.auto_global_angle = bpy.props.FloatProperty(
        name="auto_global_angle", default=0.0,
        description="Global UV rotation for the auto ratio approach"
    )
    bpy.types.Scene.auto_global_scale = bpy.props.FloatProperty(
        name="auto_global_scale", default=10.0, min=0.01,
        description="Global U scale for the auto ratio approach (units per repeat)"
    )

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.road_intersections
    del bpy.types.Scene.mark_ratio_threshold

    del bpy.types.Scene.auto_ratio_threshold
    del bpy.types.Scene.auto_global_angle
    del bpy.types.Scene.auto_global_scale

if __name__ == "__main__":
    register()
