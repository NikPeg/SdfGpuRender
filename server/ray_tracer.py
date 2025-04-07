import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

class GPUTriangles:
    def __init__(self, mesh, device):
        # Извлекаем данные треугольников
        triangles = []
        normals = []
        edges1 = []
        edges2 = []

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

            triangles.append([v0, v1, v2])
            normals.append(normal)
            edges1.append(edge1)
            edges2.append(edge2)

        # Преобразуем в тензоры PyTorch
        self.triangles = torch.tensor(triangles, dtype=torch.float32, device=device)
        self.normals = torch.tensor(normals, dtype=torch.float32, device=device)
        self.edges1 = torch.tensor(edges1, dtype=torch.float32, device=device)
        self.edges2 = torch.tensor(edges2, dtype=torch.float32, device=device)
        self.v0 = self.triangles[:, 0]  # Первые вершины всех треугольников
        self.v1 = self.triangles[:, 1]  # Вторые вершины всех треугольников
        self.v2 = self.triangles[:, 2]  # Третьи вершины всех треугольников
        self.num_triangles = len(triangles)

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

def ray_triangle_intersect_batch(ray_origins, ray_directions, gpu_triangles, batch_size=1024):
    """Пакетное пересечение лучей с треугольниками на GPU с батчингом"""
    num_rays = ray_origins.shape[0]
    device = ray_origins.device

    # Подготовка результирующих массивов
    hit = torch.zeros(num_rays, dtype=torch.bool, device=device)
    t_min = torch.ones(num_rays, dtype=torch.float32, device=device) * float('inf')
    triangle_idx = torch.zeros(num_rays, dtype=torch.int64, device=device)
    normals = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)

    # Обрабатываем лучи батчами
    for ray_batch_start in range(0, num_rays, batch_size):
        ray_batch_end = min(ray_batch_start + batch_size, num_rays)
        batch_ray_o = ray_origins[ray_batch_start:ray_batch_end]
        batch_ray_d = ray_directions[ray_batch_start:ray_batch_end]

        batch_size_actual = ray_batch_end - ray_batch_start

        # Обрабатываем треугольники батчами для каждого батча лучей
        triangle_batch_size = min(1024, gpu_triangles.num_triangles)
        batch_hit = torch.zeros(batch_size_actual, dtype=torch.bool, device=device)
        batch_t_min = torch.ones(batch_size_actual, dtype=torch.float32, device=device) * float('inf')
        batch_tri_idx = torch.zeros(batch_size_actual, dtype=torch.int64, device=device)

        for tri_batch_start in range(0, gpu_triangles.num_triangles, triangle_batch_size):
            tri_batch_end = min(tri_batch_start + triangle_batch_size, gpu_triangles.num_triangles)

            # Получаем батч треугольников
            v0_batch = gpu_triangles.v0[tri_batch_start:tri_batch_end]
            edge1_batch = gpu_triangles.edges1[tri_batch_start:tri_batch_end]
            edge2_batch = gpu_triangles.edges2[tri_batch_start:tri_batch_end]

            # Делаем broadcast для текущих батчей
            ray_o = batch_ray_o.unsqueeze(1)  # [batch_rays, 1, 3]
            ray_d = batch_ray_d.unsqueeze(1)  # [batch_rays, 1, 3]
            v0 = v0_batch.unsqueeze(0)  # [1, batch_tris, 3]
            edge1 = edge1_batch.unsqueeze(0)  # [1, batch_tris, 3]
            edge2 = edge2_batch.unsqueeze(0)  # [1, batch_tris, 3]

            # Вычисляем пересечения (алгоритм Möller–Trumbore)
            h = torch.cross(ray_d, edge2, dim=-1)  # [batch_rays, batch_tris, 3]
            a = torch.sum(edge1 * h, dim=-1)  # [batch_rays, batch_tris]

            # Игнорируем параллельные лучи и треугольники
            valid = torch.abs(a) > 1e-6  # [batch_rays, batch_tris]

            f = torch.where(valid, 1.0 / a, torch.zeros_like(a))
            s = ray_o - v0  # [batch_rays, batch_tris, 3]
            u = f * torch.sum(s * h, dim=-1)  # [batch_rays, batch_tris]

            # Обновляем валидность
            valid = valid & (u >= 0.0) & (u <= 1.0)

            q = torch.cross(s, edge1, dim=-1)  # [batch_rays, batch_tris, 3]
            v = f * torch.sum(ray_d * q, dim=-1)  # [batch_rays, batch_tris]

            # Обновляем валидность
            valid = valid & (v >= 0.0) & (u + v <= 1.0)

            # Вычисляем t
            t = f * torch.sum(edge2 * q, dim=-1)  # [batch_rays, batch_tris]
            valid = valid & (t > 0.0)

            # Маскируем невалидные результаты
            t_masked = torch.where(
                valid & torch.isfinite(t),
                t,
                torch.tensor(float('inf'), device=device)
            )

            # Находим ближайшие пересечения для текущего батча треугольников
            batch_t, batch_idx = torch.min(t_masked, dim=1)

            # Обновляем результаты, если нашли более близкие пересечения
            closer_hit = batch_t < batch_t_min
            batch_hit[closer_hit] = True
            batch_t_min[closer_hit] = batch_t[closer_hit]
            batch_tri_idx[closer_hit] = batch_idx[closer_hit] + tri_batch_start

        # Теперь у нас есть результаты для текущего батча лучей
        # Обновляем общие результаты
        hit[ray_batch_start:ray_batch_end] = batch_hit
        t_min[ray_batch_start:ray_batch_end] = batch_t_min
        triangle_idx[ray_batch_start:ray_batch_end] = batch_tri_idx

    # Заполняем нормали только для тех лучей, которые пересеклись с треугольниками
    for i in range(0, num_rays, batch_size):
        end = min(i + batch_size, num_rays)
        batch_hit = hit[i:end]
        if batch_hit.any():
            batch_idx = triangle_idx[i:end][batch_hit]
            normals[i:end][batch_hit] = gpu_triangles.normals[batch_idx]

    return hit, t_min, triangle_idx, normals


def ray_plane_intersect_batch(ray_origins, ray_directions, plane_normal, plane_d, batch_size=1024):
    """Пакетное пересечение лучей с плоскостью на GPU с батчингом"""
    num_rays = ray_origins.shape[0]
    device = ray_origins.device

    # Подготовка результирующих массивов
    hit = torch.zeros(num_rays, dtype=torch.bool, device=device)
    t = torch.ones(num_rays, dtype=torch.float32, device=device) * float('inf')

    # Обрабатываем лучи батчами
    for i in range(0, num_rays, batch_size):
        end = min(i + batch_size, num_rays)
        batch_ray_o = ray_origins[i:end]
        batch_ray_d = ray_directions[i:end]

        # Вычисляем знаменатель
        denom = torch.sum(plane_normal.unsqueeze(0) * batch_ray_d, dim=1)

        # Проверка параллельности
        valid = torch.abs(denom) > 1e-6

        # Вычисляем значения t для валидных лучей
        batch_t = torch.ones_like(denom) * float('inf')
        if valid.any():
            numer = -(torch.sum(plane_normal.unsqueeze(0) * batch_ray_o, dim=1) + plane_d)
            t_values = numer[valid] / denom[valid]
            batch_t[valid] = t_values

        # Проверка на положительность t
        batch_hit = valid & (batch_t > 0.0) & (batch_t < float('inf'))

        # Сохраняем результаты для текущего батча
        hit[i:end] = batch_hit
        t[i:end] = batch_t

    return hit, t

def reflect_ray_batch(ray_directions, normals, batch_size=1024):
    """Пакетное отражение лучей на GPU с батчингом"""
    num_rays = ray_directions.shape[0]
    device = ray_directions.device

    # Подготовка результирующего массива
    reflected = torch.zeros_like(ray_directions)

    # Обрабатываем лучи батчами
    for i in range(0, num_rays, batch_size):
        end = min(i + batch_size, num_rays)
        batch_ray_d = ray_directions[i:end]
        batch_normals = normals[i:end]

        dot_product = torch.sum(batch_ray_d * batch_normals, dim=1, keepdim=True)
        batch_reflected = batch_ray_d - 2.0 * dot_product * batch_normals

        reflected[i:end] = batch_reflected

    return reflected

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
        self.gpu_triangles = None
        self.bvh = None  # Оставляем для совместимости
        self.triangles = None  # Оставляем для совместимости

    def load_model(self, mesh):
        self.mesh = mesh

        # Создаем GPU-представление треугольников вместо BVH
        print(f"Building GPU triangles for mesh with {len(mesh.faces)} triangles...")
        self.gpu_triangles = GPUTriangles(mesh, self.device)

        # Оставляем старые структуры для совместимости
        self.bvh = BVH()
        self.bvh.build(mesh)
        self.triangles = precompute_triangles(mesh)

        return True

    def render_gpu(self, camera, width, height, max_bounces=2):
        """Рендеринг сцены с использованием GPU и батчинга"""
        aspect = width / height
        half_height = np.tan(camera['fov'] * 0.5)
        half_width = aspect * half_height

        # Преобразование параметров камеры в тензоры
        cam_pos = torch.tensor(camera['pos'], dtype=torch.float32, device=self.device)
        cam_forward = torch.tensor(camera['forward'], dtype=torch.float32, device=self.device)
        cam_right = torch.tensor(camera['right'], dtype=torch.float32, device=self.device)
        cam_up = torch.tensor(camera['up'], dtype=torch.float32, device=self.device)

        # Определение параметров сцены
        light_dir = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        light_dir = light_dir / torch.norm(light_dir)

        # Зеркальная плоскость (y = -1)
        plane_normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        plane_d = torch.tensor(1.0, dtype=torch.float32, device=self.device)  # у = -1

        background_color = torch.tensor([0.2, 0.2, 0.3], dtype=torch.float32, device=self.device)

        # Создаем буфер для результата
        result = np.zeros((height, width, 3), dtype=np.float32)

        # Разбиваем изображение на тайлы для обработки
        tile_size = 64  # Размер тайла

        for ty in range(0, height, tile_size):
            for tx in range(0, width, tile_size):
                # Определяем размеры текущего тайла
                th = min(tile_size, height - ty)
                tw = min(tile_size, width - tx)

                print(f"Processing tile at ({tx}, {ty}) with size {tw}x{th}")

                # Генерация координат пикселей для текущего тайла
                x_coords = torch.linspace(tx, tx + tw - 1, tw, device=self.device)
                y_coords = torch.linspace(ty, ty + th - 1, th, device=self.device)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

                # Преобразование координат пикселей в координаты изображения [-1, 1]
                u = (2.0 * (xx + 0.5) / width - 1.0) * half_width
                v = (1.0 - 2.0 * (yy + 0.5) / height) * half_height

                # Создание направлений лучей для тайла
                ray_directions = torch.zeros((th, tw, 3), device=self.device)
                ray_directions[..., 0] = u * cam_right[0] + v * cam_up[0] + cam_forward[0]
                ray_directions[..., 1] = u * cam_right[1] + v * cam_up[1] + cam_forward[1]
                ray_directions[..., 2] = u * cam_right[2] + v * cam_up[2] + cam_forward[2]

                # Нормализация направлений
                ray_norm = torch.norm(ray_directions, dim=-1, keepdim=True)
                ray_directions = ray_directions / ray_norm

                # Подготавливаем массивы для хранения результатов
                ray_origins = cam_pos.expand(th, tw, 3)
                pixel_colors = torch.zeros((th, tw, 3), device=self.device)
                throughput = torch.ones((th, tw, 3), device=self.device)

                # Маска активных лучей
                active_rays = torch.ones((th, tw), dtype=torch.bool, device=self.device)

                # Преобразуем массивы в 1D (один луч на пиксель)
                flat_ray_origins = ray_origins.reshape(-1, 3)
                flat_ray_directions = ray_directions.reshape(-1, 3)
                flat_throughput = throughput.reshape(-1, 3)
                flat_active_rays = active_rays.reshape(-1)
                flat_pixel_colors = pixel_colors.reshape(-1, 3)

                # Основной цикл трассировки
                for bounce in range(max_bounces):
                    print(f"Bounce {bounce}, active rays: {torch.sum(flat_active_rays).item()}")
                    # Используем только активные лучи
                    active_indices = torch.where(flat_active_rays)[0]

                    if active_indices.shape[0] == 0:
                        break

                    # Выбираем только активные лучи для вычислений
                    batch_ray_origins = flat_ray_origins[active_indices]
                    batch_ray_directions = flat_ray_directions[active_indices]
                    batch_throughput = flat_throughput[active_indices]

                    # Пересечение с мешем (используем батчинг)
                    hit_mesh, t_mesh, triangle_idx, normals = ray_triangle_intersect_batch(
                        batch_ray_origins, batch_ray_directions, self.gpu_triangles)

                    # Пересечение с плоскостью
                    hit_plane, t_plane = ray_plane_intersect_batch(
                        batch_ray_origins, batch_ray_directions, plane_normal, plane_d)

                    # Определяем, какое пересечение ближе
                    hit_mesh_closer = hit_mesh & (~hit_plane | (t_mesh < t_plane))
                    hit_plane_closer = hit_plane & (~hit_mesh | (t_plane < t_mesh))

                    print(f"Mesh hits: {torch.sum(hit_mesh).item()}, Plane hits: {torch.sum(hit_plane).item()}")
                    print(f"Mesh closer: {torch.sum(hit_mesh_closer).item()}, Plane closer: {torch.sum(hit_plane_closer).item()}")

                    # Обрабатываем попадания в меш
                    if hit_mesh_closer.any():
                        # Индексы для лучей, попавших в меш
                        mesh_hit_indices = active_indices[hit_mesh_closer]

                        # Вычисляем освещение
                        mesh_normals = normals[hit_mesh_closer]
                        dot_product = torch.sum(mesh_normals * light_dir.unsqueeze(0), dim=1).clamp(min=0.0)
                        diffuse_color = torch.tensor([0.8, 0.8, 0.8], device=self.device).unsqueeze(0)
                        ambient_color = torch.tensor([0.1, 0.1, 0.1], device=self.device).unsqueeze(0)

                        # Вычисляем цвет
                        hit_colors = batch_throughput[hit_mesh_closer] * (diffuse_color * dot_product.unsqueeze(1) + ambient_color)

                        # Обновляем цвета попаданий
                        flat_pixel_colors[mesh_hit_indices] += hit_colors

                        # Деактивируем лучи, попавшие в меш
                        flat_active_rays[mesh_hit_indices] = False

                    # Обрабатываем попадания в плоскость
                    if hit_plane_closer.any():
                        # Индексы для лучей, попавших в плоскость
                        plane_hit_indices = active_indices[hit_plane_closer]

                        # Вычисляем точки пересечения
                        hit_points = batch_ray_origins[hit_plane_closer] + batch_ray_directions[hit_plane_closer] * t_plane[hit_plane_closer].unsqueeze(1)

                        # Вычисляем отраженные лучи
                        reflected_directions = reflect_ray_batch(
                            batch_ray_directions[hit_plane_closer],
                            plane_normal.expand(hit_plane_closer.sum(), 3)
                        )

                        # Обновляем лучи для следующего отскока
                        offset_points = hit_points + 0.001 * plane_normal.unsqueeze(0)
                        flat_ray_origins[plane_hit_indices] = offset_points
                        flat_ray_directions[plane_hit_indices] = reflected_directions

                        # Уменьшаем throughput для отраженных лучей
                        flat_throughput[plane_hit_indices] *= 0.8

                    # Обрабатываем промахи
                    missed = ~(hit_mesh_closer | hit_plane_closer)
                    if missed.any():
                        miss_indices = active_indices[missed]

                        # Добавляем цвет фона
                        flat_pixel_colors[miss_indices] += flat_throughput[miss_indices] * background_color

                        # Деактивируем лучи, которые промахнулись
                        flat_active_rays[miss_indices] = False

                # Преобразуем результаты обратно в формат тайла
                pixel_colors = flat_pixel_colors.reshape(th, tw, 3)

                # Передаем результат на CPU и сохраняем в общий результат
                tile_result = pixel_colors.cpu().numpy().clip(0, 1)
                result[ty:ty+th, tx:tx+tw] = tile_result

                # Очищаем кэш CUDA для освобождения памяти
                torch.cuda.empty_cache()

        for ty in range(0, height, tile_size):
            for tx in range(0, width, tile_size):
                th = min(tile_size, height - ty)
                tw = min(tile_size, width - tx)

                # Рисуем рамку вокруг тайла
                if ty > 0:
                    result[ty, tx:tx+tw] = [1.0, 0.0, 0.0]  # Красная верхняя граница
                if ty+th < height:
                    result[ty+th-1, tx:tx+tw] = [1.0, 0.0, 0.0]  # Красная нижняя граница
                if tx > 0:
                    result[ty:ty+th, tx] = [1.0, 0.0, 0.0]  # Красная левая граница
                if tx+tw < width:
                    result[ty:ty+th, tx+tw-1] = [1.0, 0.0, 0.0]  # Красная правая граница
        return result

    def render(self, camera, width, height, max_bounces=2):
        # Используем GPU-рендеринг вместо CPU-версии
        if str(self.device) != "cpu":
            print("Using GPU accelerated rendering")
            return self.render_gpu(camera, width, height, max_bounces)
        else:
            print("Using CPU rendering (No GPU available)")
            # Вызываем оригинальный CPU метод
            return self.render_cpu(camera, width, height, max_bounces)

    def render_cpu(self, camera, width, height, max_bounces=2):
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
