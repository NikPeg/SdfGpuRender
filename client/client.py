import grpc
import pygame
import numpy as np
import sys
import time
import os
import io

# Импортируем сгенерированные gRPC классы
import sys
sys.path.append('..')
from proto import renderer_pb2
from proto import renderer_pb2_grpc

def init_camera():
    return {
        'pos': np.array([0.0, 0.0, -3.0], dtype=np.float32),
        'forward': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        'right': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'up': np.array([0.0, 1.0, 0.0], dtype=np.float32),
        'fov': np.radians(60.0)
    }

def camera_to_proto(camera):
    return renderer_pb2.CameraParams(
        position=camera['pos'].tolist(),
        forward=camera['forward'].tolist(),
        right=camera['right'].tolist(),
        up=camera['up'].tolist(),
        fov=float(camera['fov'])
    )

def load_model(stub, model_path):
    try:
        if model_path.startswith("server:"):
            # Модель на сервере
            server_path = model_path[7:]
            response = stub.LoadModel(renderer_pb2.LoadModelRequest(model_path=server_path))
        else:
            # Загружаем файл и отправляем его на сервер
            with open(model_path, 'rb') as f:
                model_data = f.read()
            response = stub.LoadModel(renderer_pb2.LoadModelRequest(model_data=model_data))

        print(f"Load model response: {response.message}")
        return response.success
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def render_frame(stub, camera, width, height, max_bounces=2):
    try:
        request = renderer_pb2.RenderRequest(
            camera=camera_to_proto(camera),
            width=width,
            height=height,
            max_bounces=max_bounces
        )

        response = stub.RenderFrame(request)

        if not response.success:
            print(f"Error from server: {response.message}")
            return None

        # Преобразуем полученные данные в NumPy массив
        pixels = np.frombuffer(response.image_data, dtype=np.uint8)
        pixels = pixels.reshape(response.height, response.width, 3)

        return pixels
    except Exception as e:
        print(f"Error rendering frame: {str(e)}")
        return None

def main(server_address, model_path):
    # Параметры окна
    width, height = 800, 600

    # Инициализация PyGame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ray Tracer Client - OBJ Viewer")
    clock = pygame.time.Clock()

    # Создаем шрифт для отображения статуса
    font = pygame.font.SysFont(None, 24)

    # Подключение к серверу
    print(f"Connecting to server at {server_address}...")
    try:
        channel = grpc.insecure_channel(server_address)
        stub = renderer_pb2_grpc.RayTracerStub(channel)
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")
        pygame.quit()
        return

    # Загрузка модели
    if not load_model(stub, model_path):
        print("Failed to load model. Exiting.")
        pygame.quit()
        return

    # Инициализация камеры
    camera = init_camera()

    # Параметры управления
    move_speed = 0.1
    rot_speed = 0.02

    # Буфер для отрисовки
    surface = pygame.Surface((width, height))

    running = True
    needs_update = True
    fps_counter = 0
    last_update_time = time.time()
    fps = 0

    # Главный цикл
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Обработка клавиш
        keys = pygame.key.get_pressed()
        movement = False

        if keys[pygame.K_w]:
            camera['pos'] += camera['forward'] * move_speed
            movement = True
        if keys[pygame.K_s]:
            camera['pos'] -= camera['forward'] * move_speed
            movement = True
        if keys[pygame.K_a]:
            camera['pos'] -= camera['right'] * move_speed
            movement = True
        if keys[pygame.K_d]:
            camera['pos'] += camera['right'] * move_speed
            movement = True
        if keys[pygame.K_UP]:
            camera['forward'] = camera['forward'] + rot_speed * camera['up']
            camera['forward'] = camera['forward'] / np.linalg.norm(camera['forward'])
            camera['right'] = np.cross(camera['forward'], np.array([0, 1, 0]))
            camera['right'] = camera['right'] / np.linalg.norm(camera['right'])
            camera['up'] = np.cross(camera['right'], camera['forward'])
            camera['up'] = camera['up'] / np.linalg.norm(camera['up'])
            movement = True
        if keys[pygame.K_DOWN]:
            camera['forward'] = camera['forward'] - rot_speed * camera['up']
            camera['forward'] = camera['forward'] / np.linalg.norm(camera['forward'])
            camera['right'] = np.cross(camera['forward'], np.array([0, 1, 0]))
            camera['right'] = camera['right'] / np.linalg.norm(camera['right'])
            camera['up'] = np.cross(camera['right'], camera['forward'])
            camera['up'] = camera['up'] / np.linalg.norm(camera['up'])
            movement = True
        if keys[pygame.K_LEFT]:
            camera['forward'] = camera['forward'] - rot_speed * camera['right']
            camera['forward'] = camera['forward'] / np.linalg.norm(camera['forward'])
            camera['right'] = np.cross(camera['forward'], np.array([0, 1, 0]))
            camera['right'] = camera['right'] / np.linalg.norm(camera['right'])
            camera['up'] = np.cross(camera['right'], camera['forward'])
            camera['up'] = camera['up'] / np.linalg.norm(camera['up'])
            movement = True
        if keys[pygame.K_RIGHT]:
            camera['forward'] = camera['forward'] + rot_speed * camera['right']
            camera['forward'] = camera['forward'] / np.linalg.norm(camera['forward'])
            camera['right'] = np.cross(camera['forward'], np.array([0, 1, 0]))
            camera['right'] = camera['right'] / np.linalg.norm(camera['right'])
            camera['up'] = np.cross(camera['right'], camera['forward'])
            camera['up'] = camera['up'] / np.linalg.norm(camera['up'])
            movement = True

        if movement:
            needs_update = True

        # Отправляем запрос на рендеринг, если требуется обновление
        if needs_update:
            print("Requesting render from server...")
            start_time = time.time()
            pixels = render_frame(stub, camera, width, height)
            end_time = time.time()

            if pixels is not None:
                # Преобразование массива NumPy в формат PyGame
                pygame_surface = pygame.surfarray.make_surface(
                    np.transpose(pixels, (1, 0, 2))
                )
                surface.blit(pygame_surface, (0, 0))

                # Обновляем счетчик FPS
                fps_counter += 1
                if time.time() - last_update_time > 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    last_update_time = time.time()

            needs_update = False

        # Отображение результата
        screen.blit(surface, (0, 0))

        # Отображение FPS и информации о камере
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        pos_text = font.render(f"Pos: {camera['pos']}", True, (255, 255, 255))
        screen.blit(pos_text, (10, 40))

        dir_text = font.render(f"Dir: {camera['forward']}", True, (255, 255, 255))
        screen.blit(dir_text, (10, 70))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client.py server_address model_path")
        print("Example: python client.py localhost:2230 ../path/to/model.obj")
        print("         python client.py localhost:2230 server:/path/on/server/model.obj")
        sys.exit(1)

    server_address = sys.argv[1]
    model_path = sys.argv[2]

    main(server_address, model_path)
