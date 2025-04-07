import grpc
import numpy as np
import trimesh
import io
import os
import time
import concurrent.futures
from concurrent import futures

# Импортируем сгенерированные gRPC классы
import sys
sys.path.append('..')
from proto import renderer_pb2
from proto import renderer_pb2_grpc

# Импортируем наш трассировщик лучей
from ray_tracer import GPURayTracer

class RayTracerServicer(renderer_pb2_grpc.RayTracerServicer):
    def __init__(self):
        self.tracer = GPURayTracer()

    def LoadModel(self, request, context):
        try:
            if request.model_path:
                # Загрузка модели из файла на сервере
                if os.path.exists(request.model_path):
                    mesh = trimesh.load(request.model_path)
                else:
                    return renderer_pb2.LoadModelResponse(
                        success=False,
                        message=f"File {request.model_path} not found on server"
                    )
            else:
                # Загрузка модели из переданных данных
                model_data = io.BytesIO(request.model_data)
                mesh = trimesh.load(model_data, file_type='obj')

            print(f"Loading mesh with {len(mesh.faces)} triangles...")
            success = self.tracer.load_model(mesh)

            return renderer_pb2.LoadModelResponse(
                success=success,
                message=f"Model loaded successfully with {len(mesh.faces)} triangles"
            )
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return renderer_pb2.LoadModelResponse(
                success=False,
                message=f"Error loading model: {str(e)}"
            )

    def RenderFrame(self, request, context):
        try:
            # Извлекаем параметры камеры
            camera = {
                'pos': np.array(request.camera.position, dtype=np.float32),
                'forward': np.array(request.camera.forward, dtype=np.float32),
                'right': np.array(request.camera.right, dtype=np.float32),
                'up': np.array(request.camera.up, dtype=np.float32),
                'fov': request.camera.fov
            }

            width = request.width
            height = request.height
            max_bounces = request.max_bounces

            # Выполняем рендеринг
            start_time = time.time()
            pixels = self.tracer.render(camera, width, height, max_bounces)
            end_time = time.time()
            print(f"Rendering completed in {end_time - start_time:.2f} seconds")

            # Подготовка данных для отправки
            pixels_uint8 = (pixels * 255).astype(np.uint8)
            image_data = pixels_uint8.tobytes()

            return renderer_pb2.RenderResponse(
                success=True,
                message=f"Rendered in {end_time - start_time:.2f} seconds",
                image_data=image_data,
                width=width,
                height=height
            )
        except Exception as e:
            print(f"Error rendering frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return renderer_pb2.RenderResponse(
                success=False,
                message=f"Error rendering frame: {str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    renderer_pb2_grpc.add_RayTracerServicer_to_server(
        RayTracerServicer(), server)
    server.add_insecure_port('[::]:2230')
    server.start()
    print("Server started, listening on port 2230")
    try:
        while True:
            time.sleep(86400) # Один день в секундах
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
