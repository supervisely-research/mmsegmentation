from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import uvicorn
import threading
import asyncio

from .request_queue import RequestQueue, RequestType


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class COCOAnnotation(BaseModel):
    bbox: List[float] = Field(..., description="[x, y, width, height]")
    category_id: int
    segmentation: Optional[List] = None
    iscrowd: int = 0
    
    @validator('bbox')
    def validate_bbox(cls, v):
        if len(v) != 4 or any(x < 0 for x in v):
            raise ValueError("bbox must be [x, y, width, height] with non-negative values")
        return v


class AddSampleRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    annotations: List[COCOAnnotation]
    filename: Optional[str] = None


def create_api(request_queue: RequestQueue) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Online Training API",
        description="API for real-time model inference and dataset updates"
    )
    
    @app.post("/predict", summary="Run inference")
    async def predict(request: PredictRequest):
        """Run model inference on an image."""
        future = request_queue.put(
            RequestType.PREDICT,
            {'image': request.image}
        )
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(503, "Request timeout - training may be busy")
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {str(e)}")
    
    @app.post("/add_sample", summary="Add training sample")
    async def add_sample(request: AddSampleRequest):
        """Add a new image and annotations to the training dataset."""
        future = request_queue.put(
            RequestType.ADD_SAMPLE,
            {
                'image': request.image,
                'annotations': [ann.dict() for ann in request.annotations],
                'filename': request.filename
            }
        )
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(503, "Request timeout - training may be busy")
        except Exception as e:
            raise HTTPException(500, f"Add sample failed: {str(e)}")
    
    @app.get("/status", summary="Get status")
    async def status():
        """Check if there are pending requests."""
        return {
            'status': 'running',
            'pending_requests': not request_queue.is_empty()
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {'status': 'healthy'}
    
    return app


def start_api_server(
    request_queue: RequestQueue,
    host: str = "0.0.0.0",
    port: int = 8000
) -> threading.Thread:
    """Start FastAPI server in a daemon thread."""
    app = create_api(request_queue)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run, daemon=True, name="APIServer")
    thread.start()
    
    print(f"\n{'='*70}")
    print(f"🚀 Online Training API Server Started")
    print(f"{'='*70}")
    print(f"   📍 URL: http://{host}:{port}")
    print(f"   📡 POST /predict      - Run inference")
    print(f"   📥 POST /add_sample   - Add training sample")
    print(f"   📊 GET  /status       - Check status")
    print(f"{'='*70}\n")
    
    return thread