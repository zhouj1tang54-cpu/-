export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error',
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  isComplete: boolean;
  timestamp: number;
}

export interface AudioVisualizerData {
  volume: number; // 0-1
}

export interface SavedSession {
  id: string;
  timestamp: number;
  preview: string;
  messages: ChatMessage[];
}

export interface UserProfile {
  name: string;
  age: string;
  avatar: string; // Emoji character
}