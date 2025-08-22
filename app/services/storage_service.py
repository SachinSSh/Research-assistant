import json
import asyncio
import aiosqlite
from datetime import datetime
from typing import Optional, List
from ..models.schemas import UserHistory, FinalBrief
from ..config import settings


class StorageService:
    def __init__(self):
        self.db_path = settings.database_url.replace("sqlite:///", "")

    async def initialize(self):
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_histories (
                    user_id TEXT PRIMARY KEY,
                    briefs TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS brief_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    token_usage TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_histories (user_id)
                )
            """)
            
            await db.commit()

    async def get_user_history(self, user_id: str) -> Optional[UserHistory]:
        """Retrieve user's research history."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT briefs, created_at, updated_at FROM user_histories WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                briefs_json, created_at, updated_at = row
                briefs_data = json.loads(briefs_json)
                
                briefs = [FinalBrief.model_validate(brief) for brief in briefs_data]
                
                return UserHistory(
                    user_id=user_id,
                    briefs=briefs,
                    created_at=datetime.fromisoformat(created_at),
                    updated_at=datetime.fromisoformat(updated_at)
                )

    async def save_user_brief(self, user_id: str, brief: FinalBrief) -> None:
        """Save a new brief to user's history."""
        async with aiosqlite.connect(self.db_path) as db:
            history = await self.get_user_history(user_id)
            
            if history:
                history.briefs.append(brief)
                history.updated_at = datetime.utcnow()
                
                if len(history.briefs) > 50:
                    history.briefs = history.briefs[-50:]
            else:
                history = UserHistory(
                    user_id=user_id,
                    briefs=[brief]
                )
            
            await db.execute("""
                INSERT INTO brief_metadata (user_id, topic, generated_at, processing_time, token_usage)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                brief.topic,
                brief.generated_at.isoformat(),
                brief.processing_time,
                json.dumps(brief.token_usage)
            ))
            
            await db.commit()

    async def get_user_stats(self, user_id: str) -> dict:
        """Get user statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT 
                    COUNT(*) as total_briefs,
                    AVG(processing_time) as avg_processing_time,
                    SUM(json_extract(token_usage, '$.total_tokens')) as total_tokens
                FROM brief_metadata 
                WHERE user_id = ?
            """, (user_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    return {
                        "total_briefs": row[0] or 0,
                        "avg_processing_time": row[1] or 0.0,
                        "total_tokens": row[2] or 0
                    }
                return {"total_briefs": 0, "avg_processing_time": 0.0, "total_tokens": 0}

