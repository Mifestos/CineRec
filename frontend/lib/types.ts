export interface User {
  id: number;
  username: string;
  email: string;
}

export interface Movie {
  movieId: number;
  title: string;
  genres: string;
  score?: number;
}

export interface Rating {
  id: number;
  user_id: number;
  movie_id: number;
  score: number;
  created_at: string;
}

export interface Review {
  id: number;
  user_id: number;
  movie_id: number;
  text: string;
  sentiment: string | null;
  sentiment_score: number | null;
  aspects: Record<string, string> | null;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}