"""POST /recommend — Phase 0 stub.

Returns an empty candidate list; wired up for real in Phase 3 once embeddings
and the dense retrieval SQL are in place.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import RecommendRequest, RecommendResponse

router = APIRouter(tags=["recommend"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    _ = request  # retrieval logic arrives in Phase 3
    return RecommendResponse(candidates=[])
