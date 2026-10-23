"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Star, MessageSquare, ThumbsUp, ThumbsDown } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea"; 
import { Badge } from "@/components/ui/badge";

interface MovieDetails {
  id: number;
  title: string;
  genres: string;
  release_year: number | null;
}

interface Review {
  id: number;
  text: string;
  sentiment: string | null;
  aspects: Record<string, string> | null;
  created_at: string;
}

export default function MoviePage() {
  const { id } = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const movieId = Number(id);

  const [rating, setRating] = useState<number>(5);
  const [reviewText, setReviewText] = useState("");

  // 1. Получаем данные фильма
  const { data: movie, isLoading: isMovieLoading } = useQuery({
    queryKey: ["movie", movieId],
    queryFn: async () => {
      const res = await api.get<MovieDetails>(`/movies/${movieId}`);
      return res.data;
    },
  });

  // 2. Получаем отзывы
  const { data: reviews, isLoading: isReviewsLoading } = useQuery({
    queryKey: ["reviews", movieId],
    queryFn: async () => {
      const res = await api.get<Review[]>(`/reviews/${movieId}`);
      return res.data;
    },
  });

  // 3. Мутация для оценки
  const rateMutation = useMutation({
    mutationFn: (score: number) => api.post("/ratings/", { movie_id: movieId, score }),
    onSuccess: () => {
      alert("Оценка сохранена!");
      // Можно инвалидировать кэш рекомендаций, чтобы при переходе назад они обновились
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });

  // 4. Мутация для отзыва
  const reviewMutation = useMutation({
    mutationFn: (text: string) => api.post("/reviews/", { movie_id: movieId, text }),
    onSuccess: () => {
      setReviewText("");
      queryClient.invalidateQueries({ queryKey: ["reviews", movieId] });
    },
  });

  if (isMovieLoading) return <div className="p-10 text-center">Загрузка...</div>;
  if (!movie) return <div className="p-10 text-center text-destructive">Фильм не найден</div>;

  return (
    <div className="container mx-auto py-10 px-4 max-w-4xl">
      <Button variant="ghost" className="mb-6" onClick={() => router.back()}>
        <ArrowLeft className="mr-2 h-4 w-4" /> Назад
      </Button>

      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-3xl">{movie.title}</CardTitle>
          <CardDescription className="text-base">
            {movie.release_year} • {movie.genres.split("|").join(", ")}
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Форма оценки и отзыва */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Поставить оценку</CardTitle>
            </CardHeader>
            <CardContent className="flex gap-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <Button
                  key={star}
                  variant={rating >= star ? "default" : "outline"}
                  onClick={() => {
                    setRating(star);
                    rateMutation.mutate(star);
                  }}
                  className="flex-1"
                >
                  {star} <Star className="ml-1 h-4 w-4" />
                </Button>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Написать отзыв</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="review">Ваш отзыв</Label>
                <Textarea
                  id="review"
                  rows={4}
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  placeholder="Поделитесь впечатлениями о сюжете, актерах, музыке..."
                />
              </div>
              <Button
                className="w-full"
                disabled={reviewText.length < 10 || reviewMutation.isPending}
                onClick={() => reviewMutation.mutate(reviewText)}
              >
                {reviewMutation.isPending ? "Анализ..." : "Отправить"}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Список отзывов */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold flex items-center gap-2">
            <MessageSquare className="h-5 w-5" /> Отзывы
          </h3>
          
          {isReviewsLoading && <p>Загрузка отзывов...</p>}
          {reviews?.length === 0 && <p className="text-muted-foreground">Пока нет отзывов.</p>}

          {reviews?.map((review) => (
            <Card key={review.id}>
              <CardContent className="pt-4">
                <p className="mb-3">{review.text}</p>
                <div className="flex flex-wrap gap-2">
                  {review.aspects && Object.entries(review.aspects).map(([aspect, sentiment]) => (
                    <Badge
                      key={aspect}
                      variant={sentiment === "positive" ? "default" : sentiment === "negative" ? "destructive" : "secondary"}
                      className="text-xs capitalize"
                    >
                      {aspect}: {sentiment}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}