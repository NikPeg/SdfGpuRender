import numpy as np
import trimesh
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    near: float = 0.01
    far: float = 1000.0

@dataclass
class AABB:
    min_bound: np.ndarray
    max_bound: np.ndarray

    def intersect(self, ray: Ray) -> bool:
        inv_dir = 1.0 / ray.direction
        t0 = (self.min_bound - ray.origin) * inv_dir
        t1 = (self.max_bound - ray.origin) * inv_dir

        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)

        tmin_val = max(np.max(tmin), ray.near)
        tmax_val = min(np.min(tmax), ray.far)

        return tmax_val >= tmin_val

@dataclass
class BVHNode:
    bbox: AABB
    left: int = -1
    right: int = -1
    start: int = -1
    count: int = 0

@dataclass
class TriangleData:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    edge1: np.ndarray
    edge2: np.ndarray
    normal: np.ndarray

class BVH:
    def __init__(self):
        self.nodes = []
        self.triangle_indices = []

    def build(self, mesh):
        primitives = []
        for i in range(len(mesh.faces)):
            face = mesh.faces[i]
            v0 = mesh.vertices[face[0]]
            v1 = mesh.vertices[face[1]]
            v2 = mesh.vertices[face[2]]

            box_min = np.minimum(np.minimum(v0, v1), v2)
            box_max = np.maximum(np.maximum(v0, v1), v2)
            primitives.append(AABB(box_min, box_max))

        self.triangle_indices = list(range(len(primitives)))
        self._build_recursive(0, 0, len(primitives), primitives)

    def _build_recursive(self, node_idx, start, end, primitives):
        node = BVHNode(
            bbox=AABB(
                np.array([float('inf'), float('inf'), float('inf')]),
                np.array([float('-inf'), float('-inf'), float('-inf')])
            ),
            start=start,
            count=end - start
        )

        # Вычисляем AABB для текущего узла
        for i in range(start, end):
            prim = primitives[self.triangle_indices[i]]
            node.bbox.min_bound = np.minimum(node.bbox.min_bound, prim.min_bound)
            node.bbox.max_bound = np.maximum(node.bbox.max_bound, prim.max_bound)

        # Листовой узел, если треугольников мало
        if node.count <= 4:
            self.nodes.append(node)
            return len(self.nodes) - 1

        # Выбираем ось разделения по максимальной протяженности
        extent = node.bbox.max_bound - node.bbox.min_bound
        axis = np.argmax(extent)
        split_pos = node.bbox.min_bound[axis] + extent[axis] * 0.5

        # Разделяем треугольники
        mid = start
        for i in range(start, end):
            prim = primitives[self.triangle_indices[i]]
            center = (prim.min_bound + prim.max_bound) * 0.5
            if center[axis] < split_pos:
                # Меняем местами индексы
                self.triangle_indices[i], self.triangle_indices[mid] = self.triangle_indices[mid], self.triangle_indices[i]
                mid += 1

        # Если не удалось разделить, делаем листовой узел
        if mid == start or mid == end:
            self.nodes.append(node)
            return len(self.nodes) - 1

        # Создаем дочерние узлы
        self.nodes.append(node)
        current_idx = len(self.nodes) - 1

        # Создаем левый и правый узлы
        node.left = self._build_recursive(len(self.nodes), start, mid, primitives)
        node.right = self._build_recursive(len(self.nodes), mid, end, primitives)
        node.count = 0  # Не листовой узел

        return current_idx

def precompute_triangles(mesh) -> List[TriangleData]:
    triangles = []
    for face in mesh.faces:
        v0 = mesh.vertices[face[0]]
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal_length = np.linalg.norm(normal)
        if normal_length > 1e-10:
            normal = normal / normal_length
        else:
            normal = np.array([0, 1, 0])  # Default normal

        triangles.append(TriangleData(v0, v1, v2, edge1, edge2, normal))
    return triangles

def ray_triangle_intersect(ray: Ray, tri: TriangleData) -> Tuple[bool, float]:
    h = np.cross(ray.direction, tri.edge2)
    a = np.dot(tri.edge1, h)

    if abs(a) < 1e-6:
        return False, 0.0

    f = 1.0 / a
    s = ray.origin - tri.v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False, 0.0

    q = np.cross(s, tri.edge1)
    v = f * np.dot(ray.direction, q)

    if v < 0.0 or u + v > 1.0:
        return False, 0.0

    t = f * np.dot(tri.edge2, q)
    if t > ray.near and t < ray.far:
        return True, t

    return False, 0.0

def ray_plane_intersect(ray_pos, ray_dir, plane_normal, plane_d) -> Tuple[bool, float]:
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) > 1e-6:
        t = -(np.dot(plane_normal, ray_pos) + plane_d) / denom
        return t >= 0, t
    return False, 0.0

def reflect_ray(incident, normal):
    return incident - 2.0 * np.dot(incident, normal) * normal

class GPURayTracer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.mesh = None
        self.bvh = None
        self.triangles = None

    def load_model(self, mesh):
        self.mesh = mesh

        # Построение BVH
        print(f"Building BVH for mesh with {len(mesh.faces)} triangles...")
        self.bvh = BVH()
        self.bvh.build(mesh)

        # Предварительное вычисление данных треугольников
        self.triangles = precompute_triangles(mesh)
        return True

    def render(self, camera, width, height, max_bounces=2):
        # Настройка параметров рендеринга
        aspect = width / height
        half_height = np.tan(camera['fov'] * 0.5)
        half_width = aspect * half_height

        # Подготовка для GPU-рендеринга
        cam_pos = torch.tensor(camera['pos'], dtype=torch.float32, device=self.device)
        cam_forward = torch.tensor(camera['forward'], dtype=torch.float32, device=self.device)
        cam_right = torch.tensor(camera['right'], dtype=torch.float32, device=self.device)
        cam_up = torch.tensor(camera['up'], dtype=torch.float32, device=self.device)

        # Свет и плоскость
        light_dir = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        light_dir = light_dir / torch.norm(light_dir)
        plane_normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)

        # Создаем направления лучей для каждого пикселя
        result = np.zeros((height, width, 3), dtype=np.float32)

        # Трассировка лучей с использованием GPU когда возможно
        # В реальной реализации здесь должна быть полная GPU-трассировка
        # Для демонстрации используем батчи для ускорения

        batch_size = 1024  # Размер батча для обработки

        # Блочная обработка для экономии памяти GPU
        for y_start in range(0, height, batch_size):
            y_end = min(y_start + batch_size, height)
            for x_start in range(0, width, batch_size):
                x_end = min(x_start + batch_size, width)

                # Генерация направлений лучей для текущего блока
                rays = []
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        u = (2.0 * (x + 0.5) / width - 1.0) * half_width
                        v = (1.0 - 2.0 * (y + 0.5) / height) * half_height
                        direction = camera['forward'] + u * camera['right'] + v * camera['up']
                        direction = direction / np.linalg.norm(direction)
                        rays.append(Ray(camera['pos'].copy(), direction, 0.01, 1000.0))

                # Трассировка лучей
                for i, ray in enumerate(rays):
                    y = y_start + i // (x_end - x_start)
                    x = x_start + i % (x_end - x_start)

                    color = np.zeros(3)
                    throughput = np.ones(3)

                    # Алгоритм трассировки лучей
                    for bounce in range(max_bounces):
                        # Поиск пересечения с мешем
                        closest_t = ray.far
                        hit_normal = None
                        mesh_hit = False

                        # Обход BVH
                        def traverse_bvh(node_idx):
                            nonlocal closest_t, hit_normal, mesh_hit

                            if node_idx >= len(self.bvh.nodes):
                                return

                            node = self.bvh.nodes[node_idx]
                            if not node.bbox.intersect(ray):
                                return

                            if node.count > 0:
                                # Листовой узел - проверяем треугольники
                                for i in range(node.count):
                                    tri_idx = self.bvh.triangle_indices[node.start + i]
                                    hit, t = ray_triangle_intersect(ray, self.triangles[tri_idx])
                                    if hit and t < closest_t:
                                        closest_t = t
                                        hit_normal = self.triangles[tri_idx].normal
                                        mesh_hit = True
                            else:
                                # Внутренний узел - рекурсивно обходим
                                traverse_bvh(node.left)
                                traverse_bvh(node.right)

                        # Начинаем обход с корня
                        traverse_bvh(0)

                        # Проверка пересечения с плоскостью
                        plane_normal_np = np.array([0, 1, 0])
                        plane_d = 1.0
                        plane_hit, plane_t = ray_plane_intersect(ray.origin, ray.direction, plane_normal_np, plane_d)

                        # Определение ближайшего пересечения
                        if mesh_hit and (not plane_hit or closest_t < plane_t):
                            # Попадание в меш
                            hit_point = ray.origin + closest_t * ray.direction
                            diffuse_color = np.array([0.8, 0.8, 0.8])
                            light_dir_np = np.array([1, 1, 1])
                            light_dir_np = light_dir_np / np.linalg.norm(light_dir_np)
                            diff = max(0.0, np.dot(hit_normal, light_dir_np))
                            color += throughput * (diffuse_color * diff + np.array([0.1, 0.1, 0.1]))
                            break
                        elif plane_hit:
                            # Попадание в зеркальную плоскость
                            hit_point = ray.origin + plane_t * ray.direction
                            reflected_dir = reflect_ray(ray.direction, plane_normal_np)

                            # Обновляем луч для следующего отскока
                            ray.origin = hit_point + 0.001 * plane_normal_np
                            ray.direction = reflected_dir

                            # Ослабляем энергию с каждым отскоком
                            throughput *= 0.8
                        else:
                            # Попадание в фон
                            background_color = np.array([0.2, 0.2, 0.3])
                            color += throughput * background_color
                            break

                    result[y, x] = np.clip(color, 0, 1)

        return result
