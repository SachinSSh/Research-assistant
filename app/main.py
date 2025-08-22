import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import settings
from .models.schemas import BriefRequest, APIResponse, FinalBrief
from .graph.workflow import ResearchWorkflow
from .services.storage_service import StorageService
from .utils.validators import validate_request_data


workflow_instance = None
storage_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global workflow_instance, storage_service
    
    print("Initializing Research Assistant API...")
    
    storage_service = StorageService()
    await storage_service.initialize()
    
    workflow_instance = ResearchWorkflow()
    
    print("✅ Research Assistant API initialized successfully")
    yield
    
    print("Shutting down Research Assistant API...")


app = FastAPI(
    title="Research Assistant API",
    description="Context-aware research brief generator using LangGraph and LangChain",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_workflow() -> ResearchWorkflow:
    """Dependency to get workflow instance."""
    if workflow_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return workflow_instance


def get_storage() -> StorageService:
    """Dependency to get storage service instance."""
    if storage_service is None:
        raise HTTPException(status_code=503, detail="Storage service not initialized")
    return storage_service


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Research Assistant API",
        "version": "1.0.0",
        "description": "Context-aware research brief generator",
        "endpoints": {
            "brief": "/brief - POST - Generate research brief",
            "health": "/health - GET - Health check",
            "history": "/history/{user_id} - GET - User research history",
            "stats": "/stats/{user_id} - GET - User statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


@app.post("/brief", response_model=APIResponse)
async def generate_brief(
    request_data: dict,
    workflow: ResearchWorkflow = Depends(get_workflow)
):
    """Generate a research brief."""
    try:
        sanitized_data = validate_request_data(request_data)
        
        try:
            brief_request = BriefRequest.model_validate(sanitized_data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")
        
        result = await workflow.execute(brief_request)
        
        if result["success"]:
            return APIResponse(
                success=True,
                data=result["brief"].model_dump(),
                trace_id=result["trace_id"],
                processing_time=result["processing_time"]
            )
        else:
            return APIResponse(
                success=False,
                error=result["error"],
                trace_id=result.get("trace_id"),
                processing_time=result.get("processing_time")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    storage: StorageService = Depends(get_storage)
):
    """Get user's research history."""
    try:
        history = await storage.get_user_history(user_id)
        if history:
            return APIResponse(
                success=True,
                data=history.model_dump()
            )
        else:
            return APIResponse(
                success=True,
                data=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/stats/{user_id}")
async def get_user_stats(
    user_id: str,
    storage: StorageService = Depends(get_storage)
):
    """Get user statistics."""
    try:
        stats = await storage.get_user_stats(user_id)
        return APIResponse(
            success=True,
            data=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


@app.post("/resume/{thread_id}")
async def resume_execution(
    thread_id: str,
    workflow: ResearchWorkflow = Depends(get_workflow)
):
    """Resume a checkpointed execution."""
    try:
        result = await workflow.resume_execution(thread_id)
        return APIResponse(
            success=result["success"],
            data=result.get("brief"),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
