from typing import Dict, Literal, Optional, Tuple, Union
from uuid import UUID

from pydantic import BaseModel


class BaseAction(BaseModel):
    type: str


class SingleClick(BaseAction):
    type: Literal["single_click"]
    coords: Tuple[int, int]


class DoubleClick(BaseAction):
    type: Literal["double_click"]
    coords: Tuple[int, int]


class DrawBox(BaseAction):
    type: Literal["draw_box"]
    coords: Tuple[Tuple[int, int], Tuple[int, int]]


class Event(BaseModel):
    inference_state: Dict
    points: "ActionCoords"
    labels: "Labels"
    frame_idx: int
    obj_id: UUID


Action = Union[SingleClick, DoubleClick, DrawBox]

ActionCoords = Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]

Labels = Union[int, Tuple[int, int]]

Prompts = Optional[Tuple[str, ActionCoords, Labels]]


class FrameMetadata(BaseModel):
    frame_id: int
    height: int
    width: int
    channels: int
    frame_size_bytes: int
    action: Optional[Action]
    timestamp: int
