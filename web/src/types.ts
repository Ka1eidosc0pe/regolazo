// Re-exports of the generated contract types used by the renderer (never edit the generated files).
export type {
  BallState,
  CameraCalibration,
  ClockStatus,
  FrameLabels,
  KitColors,
  CrestSpec,
  MatchMeta,
  ObjectLabel,
  PlayerMeta,
  PlayerState,
  SimState,
  StreamProtocol,
  TeamMeta,
} from "./types/generated/contracts.ts";

import type { PlayerState, StreamProtocol } from "./types/generated/contracts.ts";

export type ServerMessage = StreamProtocol["server"];
export type ClientMessage = StreamProtocol["client"];
export type PlayerAnim = PlayerState["anim"];
export type TeamSide = "home" | "away";

/** A recorded or scripted clip for local playback: one MatchMeta plus consecutive SimStates. */
export interface Clip {
  meta: import("./types/generated/contracts.ts").MatchMeta;
  frames: import("./types/generated/contracts.ts").SimState[];
}
